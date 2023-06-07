import re
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Import the Twitter data
def import_tweets(path):
    # Read the CSV file
    df1 = pd.read_csv(path, encoding='latin-1')

    # Drop rows with missing label values
    df1 = df1.dropna(subset=['label'])

    # Convert the 'text' column to string type
    df1["text"] = df1["text"].astype(str)

    # Remove rows with label value 0.0
    df1 = df1[df1['label'] != 0.0]

    # Map label values to 0 and 1
    df1.loc[df1['label'] == 1.0, 'label'] = 1
    df1.loc[df1['label'] == -1.0, 'label'] = 0

    # Convert the 'label' column to int type
    df1['label'] = df1['label'].astype(int)

    # Select the first 10,000 rows
    df1 = df1.head(10000)

    return df1

# Data preprocessing and cleaning
DATASET_COLUMNS = ['target','ids','date','flag','user','text']
df = pd.read_csv("training.1600000.processed.noemoticon.csv", encoding="latin", header=None, names=DATASET_COLUMNS)
df_test = import_tweets("test.csv")

# Create the dataset
dataset = df[['text', 'target']]
dataset['target'] = dataset['target'].replace(4, 1)
dataset_test = df_test[['text', 'target']]
dataset['text'] = dataset['text'].str.lower()
dataset_test['text'] = dataset_test['text'].str.lower()

stopwords = stopwords.words('english')

# Clean stopwords
def clean_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in set(stopwords)])

dataset['text'] = dataset['text'].apply(lambda text: clean_stopwords(text))
dataset_test['text'] = dataset_test['text'].apply(lambda text: clean_stopwords(text))

# Clean username mentions
def clean_username(data):
    return re.sub('@[^\s]+',' ', data)

dataset['text'] = dataset['text'].apply(lambda x: clean_username(x))
dataset_test['text'] = dataset_test['text'].apply(lambda x: clean_username(x))

# Clean URLs
def clean_url(data):
    data = re.sub(r"((https|http|ftp)?(:\/\/)?(www\.)?)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&\/\/=]*)", ' ', data)
    return re.sub(r'/', ' / ', data)

dataset['text'] = dataset['text'].apply(lambda x: clean_url(x))
dataset_test['text'] = dataset_test['text'].apply(lambda x: clean_url(x))

# Clean repeating characters
def clean_repeating_char(text):
    return re.sub(r"(.)\1\1+", r"\1\1", text)

dataset['text'] = dataset['text'].apply(lambda x: clean_repeating_char(x))
dataset_test['text'] = dataset_test['text'].apply(lambda x: clean_repeating_char(x))

# Clean emojis
def clean_emoji(data):
    data = re.sub(r'<3', '<heart>', data)
    data = re.sub(r"[8:=;]['`\-]?[)d]+", '<smile>', data)
    data = re.sub(r"[8:=;]['`\-]?\(+", '<sad>', data)
    data = re.sub(r"[8:=;]['`\-]?[\/|l*]", '<neutral>', data)
    data = re.sub(r"[8:=;]['`\-]?p+", '<laugh>', data)
    return data

dataset['text'] = dataset['text'].apply(lambda x: clean_emoji(x))
dataset_test['text'] = dataset_test['text'].apply(lambda x: clean_emoji(x))

# Clean numbers
def clean_numbers(data):
    return re.sub('[0-9]+', '', data)

dataset['text'] = dataset['text'].apply(lambda x: clean_numbers(x))
dataset_test['text'] = dataset_test['text'].apply(lambda x: clean_numbers(x))

# Clean punctuations
def clean_punctuations(text):
    return text.translate(str.maketrans('', '', string.punctuation))

dataset['text'] = dataset['text'].apply(lambda x: clean_punctuations(x))
dataset_test['text'] = dataset_test['text'].apply(lambda x: clean_punctuations(x))

# Clean non-alphabetic characters
def clean_nonalpha(data):
    return re.sub("[^a-z0-9<>]", ' ', data)

dataset['text'] = dataset['text'].apply(lambda x: clean_nonalpha(x))
dataset_test['text'] = dataset_test['text'].apply(lambda x: clean_nonalpha(x))

# Tokenize text
dataset['text'] = dataset['text'].apply(RegexpTokenizer(r'\w+').tokenize)
dataset_test['text'] = dataset_test['text'].apply(RegexpTokenizer(r'\w+').tokenize)

# Stemming
def stem_text(data):
    return [nltk.stem.snowball.SnowballStemmer(language='english').stem(word) for word in data]

dataset['text'] = dataset['text'].apply(lambda x: stem_text(x))
dataset_test['text'] = dataset_test['text'].apply(lambda x: stem_text(x))

# Lemmatization
def lemmatize_text(data):
    return [WordNetLemmatizer().lemmatize(word) for word in data]

dataset['text'] = dataset['text'].apply(lambda x: lemmatize_text(x))
dataset_test['text'] = dataset_test['text'].apply(lambda x: lemmatize_text(x))

X = dataset.text
y = dataset.target
X_test = dataset_test.text
y_test = dataset_test.target

X = X.apply(lambda x: " ".join(x))
X_test = X_test.apply(lambda x: " ".join(x))

X_train = X
y_train = y

df_train = pd.concat([X, y], axis=1)
df_train.columns = ['text', 'label']

df_test = pd.concat([X_test, y_test], axis=1)
df_test.columns = ['text', 'label']

# Save the processed variables
with open('preprocessed_data.pickle', 'wb') as f:
    pickle.dump((X, y, X_test, y_test, df_train, df_test), f)
    
print(df_train)
