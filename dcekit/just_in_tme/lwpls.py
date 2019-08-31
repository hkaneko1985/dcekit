# -*- coding: utf-8 -*-
# %reset -f
"""
@author: Hiromasa Kaneko
"""
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, RegressorMixin
#from sklearn.utils.estimator_checks import check_estimator


class LWPLS(BaseEstimator, RegressorMixin):
    def __init__(self, n_components=2, lambda_in_similarity=1.0):
        """
        Locally-Weighted Partial Least Squares (LWPLS)
        
        Parameters
        ----------
        n_components: int
            number of maximum components
        lambda_in_similarity: float
            parameter in similarity matrix
        """

        self.n_components = n_components
        self.lambda_in_similarity = lambda_in_similarity

    def fit(self, X, y):
        """
        Locally-Weighted Partial Least Squares (LWPLS)
        
        Store x and y for LWPLS
    
        Parameters
        ----------
        X : numpy.array or pandas.DataFrame
            (autoscaled) m x n matrix of X-variables of training data,
            m is the number of training sammples and
            n is the number of X-variables
        y : numpy.array or pandas.DataFrame
            (autoscaled) m x 1 vector of a Y-variable of training dat
        """

        self.x = X
        self.y = y

    def predict(self, X):
        """
        Locally-Weighted Partial Least Squares (LWPLS)
        
        Predict y-values of samples using LWPLS
    
        Parameters
        ----------
        X : numpy.array or pandas.DataFrame
            k x n matrix of X-variables of test data, which is autoscaled with training data,
            and k is the number of test samples
    
        Returns
        -------
        estimated_y : numpy.array
            k x 1 vector of estimated y-values of test data
        """

        x_train = np.array(self.x)
        y_train = np.array(self.y)
        y_train = np.reshape(y_train, (len(y_train), 1))
        x_test = np.array(X)

        estimated_y = np.zeros((x_test.shape[0], self.n_components))
        distance_matrix = cdist(x_train, x_test, 'euclidean')
        for test_sample_number in range(x_test.shape[0]):
            query_x_test = x_test[test_sample_number, :]
            query_x_test = np.reshape(query_x_test, (1, len(query_x_test)))
            distance = distance_matrix[:, test_sample_number]
            similarity = np.diag(np.exp(-distance / distance.std(ddof=1) / self.lambda_in_similarity))
            #        similarity_matrix = np.diag(similarity)

            y_w = y_train.T.dot(np.diag(similarity)) / similarity.sum()
            x_w = np.reshape(x_train.T.dot(np.diag(similarity)) / similarity.sum(), (1, x_train.shape[1]))
            centered_y = y_train - y_w
            centered_x = x_train - np.ones((x_train.shape[0], 1)).dot(x_w)
            centered_query_x_test = query_x_test - x_w
            estimated_y[test_sample_number, :] += y_w
            for component_number in range(self.n_components):
                w_a = np.reshape(centered_x.T.dot(similarity).dot(centered_y) / np.linalg.norm(
                    centered_x.T.dot(similarity).dot(centered_y)), (x_train.shape[1], 1))
                t_a = np.reshape(centered_x.dot(w_a), (x_train.shape[0], 1))
                p_a = np.reshape(centered_x.T.dot(similarity).dot(t_a) / t_a.T.dot(similarity).dot(t_a),
                                 (x_train.shape[1], 1))
                q_a = centered_y.T.dot(similarity).dot(t_a) / t_a.T.dot(similarity).dot(t_a)
                t_q_a = centered_query_x_test.dot(w_a)
                estimated_y[test_sample_number, component_number:] = estimated_y[test_sample_number,
                                                                     component_number:] + t_q_a * q_a
                if component_number != self.n_components:
                    centered_x = centered_x - t_a.dot(p_a.T)
                    centered_y = centered_y - t_a * q_a
                    centered_query_x_test = centered_query_x_test - t_q_a.dot(p_a.T)
        estimated_y = estimated_y[:, -1]
        
        if np.isnan(estimated_y).any():
            estimated_y = np.ones(X.shape[0]) * y_train.mean()
        
        return estimated_y

if __name__ == '__main__':
    check_estimator(LWPLS)