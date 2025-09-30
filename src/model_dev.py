from abc import ABC, abstractmethod
import logging
from sklearn.linear_model import LinearRegression
class Model(ABC):
    '''
    abstract class for all models
    '''
    @abstractmethod
    def train(self, X_train, y_train):
        '''
        Train the model
        
        Args:
            X_train (pd.DataFrame): training features
            y_train (pd.Series): training labels
        
        Returns:
            None
        '''
        pass

class LinearRegressionModel(Model):
    '''
    Linear Regression Model
    '''

    def train(self, X_train, y_train, **kwargs):
        '''
        Train the Linear Regression model
        
        Args:
            X_train (pd.DataFrame): training features
            y_train (pd.Series): training labels
        
        Returns:
            LinearRegression: trained Linear Regression model
        '''
        try:
            reg = LinearRegression(**kwargs)
            reg.fit(X_train, y_train)
            logging.info('Model trained successfully')
            return reg
        except Exception as e:
            logging.error('Error in training model: {}'.format(e))
            raise e