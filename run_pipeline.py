from pipelines.training_pipeline import train_pipeline
from steps.config import ModelNameConfig
from zenml.client import Client

if __name__ == '__main__': # if this script is executed directly
    # run pipeline
    print(Client().activate_stack.experiment_tracker.get_tracker_uri())
    train_pipeline(data_path='data/olist_customers_dataset.csv')