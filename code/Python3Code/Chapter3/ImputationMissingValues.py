##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 3                                               #
#                                                            #
##############################################################

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Simple class to impute missing values of a single columns.
class ImputationMissingValues:

    # Impute the mean values in case if missing data.
    def impute_mean(self, dataset, col):
        dataset[col] = dataset[col].fillna(dataset[col].mean())
        return dataset

    # Impute the median values in case if missing data.
    def impute_median(self, dataset, col):
        dataset[col] = dataset[col].fillna(dataset[col].median())
        return dataset

    # Interpolate the dataset based on previous/next values..
    def impute_interpolate(self, dataset, col):
        dataset[col] = dataset[col].interpolate()
        # And fill the initial data points if needed:
        dataset[col] = dataset[col].fillna(method='bfill')
        return dataset

    def impute_model_approach(self, dataset, col):
        imp_mean = IterativeImputer(random_state=0, verbose=1, max_iter=100)
        for i in dataset.columns:
            if i == col:
                continue
            dataset = self.impute_interpolate(dataset, i)
        print(dataset.shape)
        imp_mean.fit(dataset)
        print(dataset.columns)
        X = imp_mean.transform(dataset)
        X = pd.DataFrame(X, index=dataset.index, columns=dataset.columns)
        return X