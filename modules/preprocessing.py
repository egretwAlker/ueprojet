import numpy as np
import pandas as pd
import bson
import seaborn as sns
import string
import re
from unidecode import unidecode
import itertools
import cleantext
import nltk
import unicodedata
import unidecode
from collections import Counter
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.stem.snowball import SnowballStemmer
from collections import Counter

from tqdm import tqdm
tqdm.pandas()
 
#the stemmer requires a language parameter
snow_stemmer = SnowballStemmer(language='french')
        
def preprocess(df, language):
    stopwords_ = stopwords.words(language)
    
    # on retire les articles comme :d'
    r = df.str.replace("'", ' ')
    r = df.str.lower()
    
    # retrait de ponctuation
    punc = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    for el in punc:
        r = r.str.replace(el, ' ')
    
    # on retire les nombre
    r = r.apply(reg_nb)    

    # on retire les extra spaces
    r = r.apply(reg_spaces)
    
    # on retire les mots stopwords français
    r = r.apply(stopwords_processing, stopwords=stopwords_)
    r = r.apply(unidecode.unidecode)
    return r

def preprocess_libraries(df, language, remove_duplicates, remove_stopwords_):
    r = df.str.lower()
    #r = r.str.replace("'", ' ')
    r = r.progress_apply(lambda x:remove_punctuation(x))
    if remove_stopwords_:
        STOPWORDS = set(stopwords.words('french'))
        r = r.progress_apply(lambda text: remove_stopwords(text, STOPWORDS))
    r = r.progress_apply(unidecode.unidecode)
    
    r = r.apply(reg_nb)
    r = r.apply(reg_spaces)
    
    snow_stemmer = SnowballStemmer(language='french')
    r = r.progress_apply(lambda x: stem_words(x, snow_stemmer))
    if remove_duplicates:
        r = r.progress_apply(remov_duplicates)
    return r
    
    
def get_most_common_words(df, n):
    results = Counter()
    df = df.dropna(inplace=False)
    df['comment'].str.split().apply(results.update)
    popular_words = sorted(results, key = results.get, reverse = True)[:n]
    return popular_words
    
def remov_duplicates(input):
 
    # split input string separated by space
    input = input.split(" ")
 
    # now create dictionary using counter method
    # which will have strings as key and their
    # frequencies as value
    UniqW = Counter(input)
 
    # joins two adjacent elements in iterable way
    return " ".join(UniqW.keys())
    
def stem_words(x, snow_stemmer):
    return " ".join([snow_stemmer.stem(word) for word in x.split()])
    
PUNCT_TO_REMOVE = string.punctuation
def remove_punctuation(text):
    """custom function to remove the punctuation"""
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

def remove_stopwords(text, STOPWORDS):
    """custom function to remove the stopwords"""
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

def show_popular_words(df, most_common_nb):
    df = df.astype('string')
    words = ' '.join([str(i) for i in df]).split()
    counter_words = Counter(words)
    wordcloud = WordCloud(width = 1000, height = 500).generate_from_frequencies(counter_words)
    plt.imshow(wordcloud)
    plt.show()
    return counter_words.most_common(most_common_nb)

# on retire les mots de longueur <= 3
def reg_length(x):
    try:
        return re.sub(r'\b\w{1,3}\b', '', x)
    except:
        return x
    
# on retire les whitespaces
def reg_spaces(x):
    try: 
        return re.sub(' +', ' ', x)
    except:
        return x
    
    
# on retire les nombres
def reg_nb(x):
    try:
        return re.sub('\d', ' ', x)
    except:
        return x   
    
    
def stopwords_processing(x, stopwords):
    try:
        split = x.split()
        words = [word for word in split if word not in stopwords]
        return " ".join(words)
    except Exception as e:
        print(e)
        return x
    
    

    
    
    
    
    
    
    
    
    
    
    