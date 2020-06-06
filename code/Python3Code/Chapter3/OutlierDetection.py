##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 3                                               #
#                                                            #
##############################################################

import scipy
import math
from sklearn.mixture import GaussianMixture
import numpy as np
import pandas as pd
import util.util as util
from sklearn.neighbors import NearestNeighbors
import copy

# Class for outlier detection algorithms based on some distribution of the data. They
# all consider only single points per row (i.e. one column).
class DistributionBasedOutlierDetection:

    # Finds outliers in the specified column of datatable and adds a binary column with
    # the same name extended with '_outlier' that expresses the result per data point.
    def chauvenet(self, data_table, col):
        # Taken partly from: https://www.astro.rug.nl/software/kapteyn/

        # Computer the mean and standard deviation.
        mean = data_table[col].mean()
        std = data_table[col].std()
        N = len(data_table.index)
        criterion = 1.0/(2*N)

        # Consider the deviation for the data points.
        deviation = abs(data_table[col] - mean)/std

        # Express the upper and lower bounds.
        low = -deviation/math.sqrt(2)
        high = deviation/math.sqrt(2)
        prob = []
        mask = []

        # Pass all rows in the dataset.
        for i in range(0, len(data_table.index)):
            # Determine the probability of observing the point
            prob.append(1.0 - 0.5 * (scipy.special.erf(high[i]) - scipy.special.erf(low[i])))
            # And mark as an outlier when the probability is below our criterion.
            mask.append(prob[i] < criterion)
        data_table[col + '_outlier'] = mask
        return data_table

    # Fits a mixture model towards the data expressed in col and adds a column with the probability
    # of observing the value given the mixture model.
    def mixture_model(self, data_table, col):

        print('Applying mixture models')
        # Fit a mixture model to our data.
        data = data_table[data_table[col].notnull()][col]
        g = GaussianMixture(n_components=3, max_iter=100, n_init=1)
        reshaped_data = np.array(data.values.reshape(-1,1))
        g.fit(reshaped_data)

        # Predict the probabilities
        probs = g.score_samples(reshaped_data)

        # Create the right data frame and concatenate the two.
        data_probs = pd.DataFrame(np.power(10, probs), index=data.index, columns=[col+'_mixture'])

        data_table = pd.concat([data_table, data_probs], axis=1)

        return data_table

# Class for distance based outlier detection.
class DistanceBasedOutlierDetection:

    # Create distance table between rows in the data table. Here, only cols are considered and the specified
    # distance function is used to compute the distance.
    def distance_table(self, data_table, cols, d_function):

        data_table[cols] = data_table.loc[:, cols].astype('float32')

        return pd.DataFrame(scipy.spatial.distance.squareform(util.distance(data_table.loc[:, cols], d_function)),
                            columns=data_table.index, index=data_table.index).astype('float32')

    # Create table that contain k neighbors for every data point. Only cols are considered and the specified
    # distance function is used. Also distances to these neighbors in form of array of distionaries is returned.
    def k_nearest_neighbors(self, data_table, cols, k, d_function):

        data_table_essential_columns = data_table.loc[:, cols].astype('float32')
        # k+1 because we count also distance from point to itself.
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='kd_tree', metric=d_function).fit(data_table_essential_columns)
        distances, indices = nbrs.kneighbors(data_table_essential_columns)

        neighbor_distances = [dict(zip(x,y)) for (x,y) in zip(indices,distances)]
        return neighbor_distances, indices

    # The most simple distance based algorithm. We assume a distance function, e.g. 'euclidean'
    # and a minimum distance of neighboring points and frequency of occurrence.
    def simple_distance_based(self, data_table, cols, d_function, dmin, fmin):
        print('Calculating simple distance-based criterion.')

        # Normalize the dataset first.
        new_data_table = util.normalize_dataset(data_table.dropna(axis=0, subset=cols), cols)

        # Create the distance table first between all instances:
        # self.distances = self.distance_table(new_data_table, cols, d_function)
        print(int(len(new_data_table.index) * (1 - fmin)))
        self.neighbor_distances, self.neighbors = self.k_nearest_neighbors(new_data_table, cols, int(len(new_data_table.index) * (1 - fmin)), d_function)

        mask = []
        # Pass the rows in our table.
        for i in range(0, len(new_data_table.index)):
            # Check what faction of neighbors are beyond dmin.
            # frac = (float(sum([1 for col_val in self.distances.iloc[i,:].tolist() if col_val > dmin]))/len(new_data_table.index))
            # Mark as an outlier if beyond the minimum frequency.
            # mask.append(frac > fmin)
            max_distance_to_nearest_neighbour, _ = self.k_distance(i)
            mask.append(dmin < max_distance_to_nearest_neighbour)
        data_mask = pd.DataFrame(mask, index=new_data_table.index, columns=['simple_dist_outlier'])
        data_table = pd.concat([data_table, data_mask], axis=1)
        # del self.distances
        del self.neighbors
        del self.neighbor_distances
        return data_table

    # Computes the local outlier factor. K is the number of neighboring points considered, d_function
    # the distance function again (e.g. 'euclidean').
    def local_outlier_factor(self, data_table, cols, d_function, k):
        # Inspired by https://github.com/damjankuznar/pylof/blob/master/lof.py
        # but tailored towards the distance metrics and data structures used here.

        print("Calculating local outlier factor.")

        # Normalize the dataset first.
        new_data_table = util.normalize_dataset(data_table.dropna(axis=0, subset=cols), cols)
        # Create nearest k neighbors table and distances t othem for ecery data point.
        self.neighbor_distances, self.neighbors = self.k_nearest_neighbors(new_data_table, cols, k, d_function)

        outlier_factor = []
        # Compute the outlier score per row.
        for i in range(0, len(new_data_table.index)):
            if i%100==0: print(f'Completed {i} steps for LOF.')
            outlier_factor.append(self.local_outlier_factor_instance(i, k))
        data_outlier_probs = pd.DataFrame(outlier_factor, index=new_data_table.index, columns=['lof'])
        data_table = pd.concat([data_table, data_outlier_probs], axis=1)
        del self.neighbors
        del self.neighbor_distances
        return data_table

    # The distance between a row i1 and i2.
    def reachability_distance(self, i1, i2):
        # Compute the k-distance of i2.
        k_distance_value, neighbors = self.k_distance(i2)
        # The value is the max of the k-distance of i2 and the real distance.
        return max([k_distance_value, self.neighbor_distances[i1][i2]])

    # Compute the local reachability density for a row i, given a k-distance and set of neighbors.
    def local_reachability_density(self, root_i, neighbors_i):
        # Set distances to neighbors to 0.
        reachability_distances_array = [0]*len(neighbors_i)

        # Compute the reachability distance between i and all neighbors.
        for i, neighbor in enumerate(neighbors_i):
            if neighbor == root_i:
                continue
            reachability_distances_array[i] = self.reachability_distance(root_i, neighbor)
        if not any(reachability_distances_array):
            return float(2.0)
        else:
            # Return the number of neighbors divided by the sum of the reachability distances.
            return len(neighbors_i) / sum(reachability_distances_array)

    # Compute the k-distance of a row i, namely the maximum distance within the k nearest neighbors
    # and return a tuple containing this value and the neighbors within this distance.
    def k_distance(self, i):
        neighbors = self.neighbors[i]
        k_distance_value = max(self.neighbor_distances[i].values())
        return k_distance_value, neighbors

    # Compute the local outlier score of our row i given a setting for k.
    def local_outlier_factor_instance(self, root_i, k):
        # Compute the k-distance for i.
        k_distance_value, neighbors = self.k_distance(root_i)
        # Computer the local reachability given the found k-distance and neighbors.
        instance_lrd = self.local_reachability_density(root_i, neighbors)
        lrd_ratios_array = [0] * len(neighbors)

        # Computer the k-distances and local reachability density of the neighbors
        for i, neighbor in enumerate(neighbors):
            if neighbor == root_i:
                continue
            k_distance_value_neighbor, neighbors_neighbor = self.k_distance(neighbor)
            neighbor_lrd = self.local_reachability_density(neighbor, neighbors_neighbor)

            # It may appear that both lrd values are infinity.
            # In that case it is wise to assume that their proportion is 1.
            if np.isinf(neighbor_lrd) and np.isinf(instance_lrd):
                lrd_ratios_array[i] = 1.0
            else:
                # Store the ratio between the neighbor and the row i.
                lrd_ratios_array[i] = neighbor_lrd / instance_lrd

        # Return the average ratio.
        sum_lrd_ratios = sum(lrd_ratios_array)
        # It can happen that instance_lrd is not INF, but some of the neighbors have lrd == INF.
        # In that case we assume that the lof is a very big number, lets say 1000.
        if (np.isinf(sum_lrd_ratios)):
            sum_lrd_ratios = 10.0
        return sum_lrd_ratios / len(neighbors)