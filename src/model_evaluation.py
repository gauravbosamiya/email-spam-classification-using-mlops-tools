import os
import pandas as pd
import numpy as np 
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import logging


log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("data ingestion")
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir,'model_evaluation.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_model(file_path:str):
    try:
        with open(file_path,'rb') as file:
            model = pickle.load(file)
        logger.debug("model loaded from: %s", file_path)
        return model
    except FileNotFoundError as e:
        logger.error("model file not found: %s", e)
        raise
    except Exception as e:
        logger.error("unexpected error: %s", e)
        raise
    
def load_data(file_path:str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logger.debug("Data loaded from: %s", file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error("dataframe is not loaded: %s", e)
        raise
    except Exception as e:
        logger.error("unexpected error: %s", e)
        raise

def evaluate_model(clf, X_test:np.ndarray, y_test: np.ndarray) -> dict:
    try:
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:,1]
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test,y_pred_proba)
        # conf_matrix = confusion_matrix(y_test, y_pred)
        
        
        metrics_dict = {
            'accuracy' : accuracy,
            'precision' : precision,
            'recall' : recall,
            'auc' : auc
            # 'confusion_matrix' : conf_matrix
        }
        logger.debug('Model evaluation metrics calculated')
        return metrics_dict
    
    except Exception as e:
        logger.error('error in model evealuation: %s', e)
        raise
    
def save_matrics(metrics: dict, file_path:str) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.debug("metrics saved: %s", file_path)
    except Exception as e:
        logger.error("error in saving metrics: %s", e)
        raise
    

def main():
    try:
        clf = load_model('./models/model2.pkl')
        test_data = load_data('./data/processed/test_tfidf.csv')
        
        X_test = test_data.iloc[:,:-1].values
        y_test = test_data.iloc[:,-1].values
        
        metrics = evaluate_model(clf, X_test, y_test)
        
        save_matrics(metrics, 'reports/metrics.json')
    except Exception as e:
        logger.error('model evaluation process failed: %s', e)
        print(f"Error {e}")
        
if __name__=='__main__':
    main()