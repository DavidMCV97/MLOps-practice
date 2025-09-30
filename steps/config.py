from pydantic import BaseModel

class ModelNameConfig(BaseModel):
    '''
    Model name config
    '''
    model_name: str = 'linear_regression'