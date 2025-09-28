import logging
import pandas as pd
from zenml import step

@step
def train_model(df:pd.DataFrame) -> None:
    '''
    Trains the model with the ingested data

    Args:
        df: ingested data
    '''
    pass