import os 
import numpy as np
import pandas as pd 
import pickle
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
import yaml

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("data ingestion")
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir,'model_training.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path:str) -> dict: 
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error("file not found: %s", params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('unexpected error: %s', e)
        raise

def load_data(file_path:str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logger.debug('Data loaded from %s with shape %s', file_path, df.shape)
        return df
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
    except pd.errors.EmptyDataError as e:
        logger.error('df not loaded: %s', e)
    except Exception as e:
        logger.error("unexpected error: %s", e)
        print(f"Error: {e}")
        
def train_model(X_train:np.ndarray, y_train:np.ndarray, params:dict) -> RandomForestClassifier:
    try:
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("Shape is differnet in X_train and y_train")
        
        logger.debug('intialize RandomForestClassifer')
        clf = RandomForestClassifier(n_estimators=params['n_estimators'], random_state=params['random_state'],
                                     max_depth=params['max_depth'], bootstrap=params['bootstrap'])
        
        logger.debug('Model training start with %d samples', X_train.shape[0])
        clf.fit(X_train,y_train)
        logger.debug('Model training completed')
        
        return clf
    except ValueError as e:
        logger.error('Value error durirng model training: %s', e)
        raise
    except Exception as e:
        logger.error('error during model training: %s', e)
        raise

def save_model(model, file_path:str) -> None:
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,"wb") as file:
            pickle.dump(model,file)
        logger.debug("Model saved to: %s", file_path)
    except FileNotFoundError as e:
        logger.error('File path not found: %s', e)
        raise
    except Exception as e:
        logger.error("Model not saved: %s", e)
        raise
    
def main():
    try:
        params = load_params('params.yaml')['model_training']
        # params = {'n_estimators': 25, 'random_state': 2}
        
        train_data = load_data('./data/processed/train_tfidf.csv')
        x_train = train_data.iloc[:,:-1].values
        y_train = train_data.iloc[:,-1].values
        
        clf = train_model(x_train, y_train, params)
        
        model_save_path = 'models/model2.pkl'
        save_model(clf,model_save_path)
    except Exception as e:
        logger.error('Model building process failed: %s', e)
        print(f"Error: {e}")

if __name__=='__main__':
    main()
    