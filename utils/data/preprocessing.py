import re
import contractions
import string
import nltk
from nltk.corpus import stopwords
from cleantext import clean
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def remove_timestamps(text):
    # Regular expression to match timestamps like "04:21, 14 May 2007"
    text = re.sub(r'\d{2}:\d{2},?\s*(\d{1,2}\s*[a-zA-Z]{3,16}|\s*[a-zA-Z]{3,16}\s*\d{1,2}),?\s*\d{4}', '', text)
    text = re.sub(r'\d{1,2} [a-zA-Z]{3,16}', '', text)
    text = re.sub(r'\d{2}:\d{2}', '', text)

    return text

def preprocess_text(text):
    if isinstance(text, str):
        
        text = re.sub(r'\bUTC\b', '', text)
        text = re.sub(r'==+', '', text)
        text = re.sub(r'\|', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    return text

def clean_text_for_bert(text, remove_numbers=False, expand_contractions=True, remove_stopwords=True, lemmatize=True):
    if isinstance(text, str):

        text = remove_timestamps(text)
        
        if expand_contractions:
            text = contractions.fix(text)

        text = preprocess_text(text)
        
        text = clean(text,
                     fix_unicode=True,
                     to_ascii=True,
                     lower=True,
                     no_line_breaks=True,
                     no_urls=True,
                     no_emails=True,
                     no_phone_numbers=True,
                     no_numbers=remove_numbers,
                     no_digits=remove_numbers,
                     no_currency_symbols=True,
                     no_punct=True,
                     replace_with_url="",
                     replace_with_email="",
                     replace_with_phone_number="",
                     replace_with_number="",         
                     replace_with_digit="",          
                     replace_with_currency_symbol="",
                     lang="en")                      
        
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        text = re.sub(r'\s+', ' ', text).strip()

        if lemmatize:
            text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

        text = text.strip()

        return text
    return text

def deduplicate_rows(df, text_column='cleaned_text'):
    """
    This function removes duplicate rows based on the cleaned_text column.
    
    Args:
    df (pd.DataFrame): The input DataFrame.
    text_column (str): The name of the text column to check for duplicates (default: 'cleaned_text').
    
    Returns:
    pd.DataFrame: A deduplicated DataFrame.
    """
    print(f"Initial DataFrame shape: {df.shape}")
    
    df_dedup = df.drop_duplicates(subset=text_column, keep='first')
    
    print(f"DataFrame shape after deduplication: {df_dedup.shape}")
    
    return df_dedup