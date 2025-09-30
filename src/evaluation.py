import logging
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

class Evaluation(ABC):
    '''
    Abstract class for model evaluation
    '''
    
    @abstractmethod
    def calculate_scores(self, y_true:np.ndarray, y_pred:np.ndarray):
        '''
        Calculate evaluation scores
        
        Args:
            y_true (np.ndarray): true labels
            y_pred (np.ndarray): predicted labels
        Returns:
            none
        '''
        pass

class MSE(Evaluation):
    '''
    Evaluation strategy using Mean Squared Error
    '''

    def calculate_scores(self, y_true:np.ndarray, y_pred:np.ndarray):
        try:
            logging.info('Calculating Mean Squared Error')
            mse = mean_squared_error(y_true, y_pred)
            logging.info(f'Mean Squared Error: {mse}')
            return mse
        except Exception as e:
            logging.error('Error in calculating MSE: {}'.format(e))
            raise e

class R2Score(Evaluation):
    '''
    Evaluation strategy using R2 Score
    '''

    def calculate_scores(self, y_true:np.ndarray, y_pred:np.ndarray):
        try:
            logging.info('Calculating R2 Score')
            r2 = r2_score(y_true, y_pred)
            logging.info(f'R2 Score: {r2}')
            return r2
        except Exception as e:
            logging.error('Error in calculating R2 Score: {}'.format(e))
            raise e

class RMSE(Evaluation):
    '''
    Evaluation strategy using Root Mean Squared Error
    '''

    def calculate_scores(self, y_true:np.ndarray, y_pred:np.ndarray):
        try:
            logging.info('Calculating Root Mean Squared Error')
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            logging.info(f'Root Mean Squared Error: {rmse}')
            return rmse
        except Exception as e:
            logging.error('Error in calculating RMSE: {}'.format(e))
            raise e