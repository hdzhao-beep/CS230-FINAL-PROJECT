# Author: [Handi Zhao/hdzhao]
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

def import_tweets(path):
    df = pd.read_csv(path, encoding='latin-1')
    df = df.dropna(subset=['label'])
    df["text"] = df["text"].astype(str)
    df = df[df['label'] != 0.0]
    df.loc[df['label'] == 1.0, 'label'] = 1
    df.loc[df['label'] == -1.0, 'label'] = 0
    df['label'] = df['label'].astype(int)
    df = df.head(10000)
    return df

def clean_stopwords(text):
    stopwords = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
                 'and', 'any', 'are', 'as', 'at', 'be', 'because', 'been', 'before',
                 'being', 'below', 'between', 'both', 'by', 'can', 'd', 'did', 'do',
                 'does', 'doing', 'down', 'during', 'each', 'few', 'for', 'from',
                 'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
                 'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
                 'into', 'is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
                 'me', 'more', 'most', 'my', 'myself', 'now', 'o', 'of', 'on', 'once',
                 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'own', 're', 's',
                 'same', 'she', "shes", 'should', "shouldve", 'so', 'some', 'such',
                 't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
                 'themselves', 'then', 'there', 'these', 'they', 'this', 'those',
                 'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was',
                 'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom',
                 'why', 'will', 'with', 'won', 'y', 'you', "youd", "youll", "youre",
                 "youve", 'your', 'yours', 'yourself', 'yourselves']
    return " ".join([word for word in str(text).split() if word not in set(stopwords)])

def clean_username(data):
    return re.sub('@[^\s]+', ' ', data)

def clean_url(data):
    data = re.sub(r"((https|http|ftp)?(:\/\/)?(www\.)?)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&\/\/=]*)", ' ', data)
    return re.sub(r'/', ' / ', data)

def clean_repeating_char(text):
    return re.sub(r"(.)\1\1+", r"\1\1", text)

def clean_emoji(data):
    data = re.sub(r'<3', '<heart>', data)
    data = re.sub(r"[8:=;]['`\-]?[)d]+", '<smile>', data)
    data = re.sub(r"[8:=;]['`\-]?\(+", '<sad>', data)
    data = re.sub(r"[8:=;]['`\-]?[\/|l*]", '<neutral>', data)
    data = re.sub(r"[8:=;]['`\-]?p+", '<laugh>', data)
    return data

def clean_numbers(data):
    return re.sub('[0-9]+', '', data)

def clean_punctuations(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def clean_nonalpha(data):
    return re.sub("[^a-z0-9<>]", ' ', data)

def stem_text(data):
    return [nltk.stem.snowball.SnowballStemmer(language='english').stem(word) for word in data]

def lemmatize_text(data):
    return [WordNetLemmatizer().lemmatize(word) for word in data]


DATASET_COLUMNS = ['target', 'ids', 'date', 'flag', 'user', 'text']
df = pd.read_csv("training.1600000.processed.noemoticon.csv", encoding="latin", header=None, names=DATASET_COLUMNS)
df_test = import_tweets("test.csv")

dataset = df[['text', 'target']]
dataset['target'] = dataset['target'].replace(4, 1)

dataset_test = df_test[['text', 'target']]
dataset_test['text'] = dataset_test['text'].str.lower()

dataset['text'] = dataset['text'].apply(lambda text: clean_stopwords(text))
dataset_test['text'] = dataset_test['text'].apply(lambda text: clean_stopwords(text))

dataset['text'] = dataset['text'].apply(lambda x: clean_username(x))
dataset_test['text'] = dataset_test['text'].apply(lambda x: clean_username(x))

dataset['text'] = dataset['text'].apply(lambda x: clean_url(x))
dataset_test['text'] = dataset_test['text'].apply(lambda x: clean_url(x))

dataset['text'] = dataset['text'].apply(lambda x: clean_repeating_char(x))
dataset_test['text'] = dataset_test['text'].apply(lambda x: clean_repeating_char(x))

dataset['text'] = dataset['text'].apply(lambda x: clean_emoji(x))
dataset_test['text'] = dataset_test['text'].apply(lambda x: clean_emoji(x))

dataset['text'] = dataset['text'].apply(lambda x: clean_numbers(x))
dataset_test['text'] = dataset_test['text'].apply(lambda x: clean_numbers(x))

dataset['text'] = dataset['text'].apply(lambda x: clean_punctuations(x))
dataset_test['text'] = dataset_test['text'].apply(lambda x: clean_punctuations(x))

dataset['text'] = dataset['text'].apply(lambda x: clean_nonalpha(x))
dataset_test['text'] = dataset_test['text'].apply(lambda x: clean_nonalpha(x))

dataset['text'] = dataset['text'].apply(RegexpTokenizer(r'\w+').tokenize)
dataset_test['text'] = dataset_test['text'].apply(RegexpTokenizer(r'\w+').tokenize)

dataset['text'] = dataset['text'].apply(lambda x: stem_text(x))
dataset_test['text'] = dataset_test['text'].apply(lambda x: stem_text(x))

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
