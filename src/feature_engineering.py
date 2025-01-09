import pandas as pd 
import os 
from sklearn.feature_extraction.text import TfidfVectorizer
import logging


log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("data ingestion")
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir,'feature_engineering.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(file_path:str)->pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        df.fillna('',inplace=True)
        logger.debug('Data Loaded and null values are filled: %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error ocurred: %s', e)
        raise
    
def apply_tfidf(train_data:pd.DataFrame, test_data:pd.DataFrame, max_features:int)->tuple:
    try:
        vectoriz = TfidfVectorizer(max_features=max_features)
        X_train = train_data['Text'].values
        y_train = train_data['Target'].values
        X_test = test_data['Text'].values
        y_test = test_data['Target'].values
        
        X_train_row = vectoriz.fit_transform(X_train)
        X_test_row = vectoriz.fit_transform(X_test)
        
        train_df = pd.DataFrame(X_train_row.toarray())
        train_df['label'] = y_train
        
        test_df = pd.DataFrame(X_test_row.toarray())
        test_df['label'] = y_test
        
        logger.debug('TfidfVectorizer applied')
        return train_df, test_df
    except Exception as e:
        logger.error('Error during TfidfVectorizer: %s', e)
        raise
    
def save_data(df:pd.DataFrame, file_path:str) -> None:
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.debug('Data saved: %s', file_path)
    except Exception as e:
        logger.error('Error while savig data: %s', e)
        raise
    
def main():
    try:
        max_features = 50
        train_data = load_data('./data/interim/train_processed.csv')
        test_data = load_data('./data/interim/test_processed.csv')
        
        train_df, test_df = apply_tfidf(train_data, test_data,max_features)
        
        save_data(train_df, os.path.join("./data","processed","train_tfidf.csv"))
        save_data(test_df, os.path.join("./data","processed","test_tfidf.csv"))
    except Exception as e:
        logger.error('Failed feature engineering: %s', e)
        print(f"Error: {e}")
        
if __name__=='__main__':
    main()