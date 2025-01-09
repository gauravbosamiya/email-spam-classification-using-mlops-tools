import os
import logging
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("data ingestion")
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir,'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def transform_test(text):
    # text = text.lower()
    # for txt in text:
    #     if txt in string.punctuation:
    #         text = text.replace(txt,"").strip()
            
    # stop_words = stopwords.words('english')
    # keep_words = []
    # for words in text.split():
    #     if words not in stop_words:
    #         keep_words.append(words)
    
    # text = nltk.word_tokenize(text)
    
    # ps = PorterStemmer()
    # lst = []
    # for i in text:
    #     lst.append(ps.stem(i))
        
    # return ' '.join(lst)

    ps = PorterStemmer()
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    text = [word for word in text if word.isalnum()]
    
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    text = [ps.stem(word) for word in text]
    return " ".join(text)


def prepocess_df(df,text_column='Text',target_column='Target'):
    try:
        logger.debug('starting preprocessing for DataFrame')
        encoder = LabelEncoder()
        df[target_column] = encoder.fit_transform(df[target_column])
        logger.debug('Target column encoded')
        
        df = df.drop_duplicates(keep='first')
        logger.debug('Duplicates removed')
        
        df.loc[:,text_column] = df[text_column].apply(transform_test)
        logger.debug('Text column transformed')
        return df
    except KeyError as e:
        logger.error('Column not found: %s', e)
        raise
    except Exception as e:
        logger.error("Error during text normalization: %s",e)
        raise
    
def main(text_column='Text',target_column='Target'):
    try:
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.debug('Data loaded properly')
        
        train_processed_data = prepocess_df(train_data,text_column,target_column)
        test_processed_data = prepocess_df(test_data,text_column,target_column)
        
        data_path = os.path.join("./data","interim")
        os.makedirs(data_path,exist_ok=True)
        
        train_processed_data.to_csv(os.path.join(data_path,"train_processed.csv"),index=False)
        test_processed_data.to_csv(os.path.join(data_path,"test_processed.csv"),index=False)
        
        logger.debug('Processed data saved to: %s', data_path)
        
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
    except pd.errors.EmptyDataError as e:
        logger.error('df not found: %s', e)
    except Exception as e:
        logger.error("Failed to complete the data transformation process: %s", e)
        print(f"Error: {e}")
        
if __name__=='__main__':
    main()