import logging
import pandas as pd
from zenml import step
from .config import ModelNameConfig
from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from zenml.client import Client
import mlflow

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config: ModelNameConfig) -> RegressorMixin:
    '''
    Trains the model with the ingested data

    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Testing features
        y_train (pd.Series): Training target
        y_test (pd.Series): Testing target
        config (ModelNameConfig): Configuration for model training
    Returns:
        RegressorMixin: Trained model
    '''
    try:
        model = None
        if config.model_name == 'linear_regression':
            mlflow.sklearn.autolog()
            model = LinearRegressionModel()
            trained_model = model.train(X_train, y_train)
            logging.info('Model training completed')
            return trained_model
        else:
            raise ValueError(f"Model {config.model_name} not supported")
    except Exception as e:
        logging.error('Error in training model: {}'.format(e))
        raise e