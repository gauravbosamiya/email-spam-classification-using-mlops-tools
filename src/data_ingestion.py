import pandas as pd 
import os 
from sklearn.model_selection import train_test_split
import logging

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("data ingestion")
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir,'data ingestion.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_data(data_url:str) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_url)
        logger.debug('Data loaded from: %s',data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df.rename(columns={'Category' : 'Target', 'Message':'Text'}, inplace=True)
        logger.debug("Data preprocessing completed")
        return df
    except KeyError as e:
        logger.error("Missing columns in the dataframe: %s", e)
        raise
    except KeyError as e:
        logger.error("Unexpected error during preprocessing: %s", e)
        raise
    
def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    try:
        raw_data_path = os.path.join(data_path,'raw')
        os.makedirs(raw_data_path,exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path,"train.csv"),index=False)
        test_data.to_csv(os.path.join(raw_data_path,"test.csv"),index=False)
        logger.debug("Train and Test data saved to: %s", raw_data_path)
    except Exception as e:
        logger.error("unexpected error occured while saving the data: %s", e)
        raise
    
def main():
    try:
        test_size = 0.2
        data_path = "https://raw.githubusercontent.com/gauravbosamiya/Datasets/refs/heads/main/email.csv"
        df = load_data(data_path)
        final_df = preprocess_data(df)
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=2)
        save_data(train_data,test_data, data_path='./data')
    except Exception as e:
        logger.error("Falied to complete the data ingestion process: %s", e)
        print(f"Error: {e}")
        
if __name__ == '__main__':
    main()