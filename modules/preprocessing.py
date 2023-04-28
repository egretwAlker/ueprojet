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
    # regex pour retirer tous les mots dont length <= 3
    r = r.apply(reg_length)
    # on retire les extra spaces
    r = r.apply(reg_spaces)
    
    # on retire les mots stopwords français
    r = r.apply(stopwords_processing, stopwords=stopwords_)
    r = r.apply(unidecode.unidecode)
    return r


def show_popular_words(df, most_common_nb):
    df = df.astype('string')
    words = ' '.join([str(i) for i in df]).split()
    counter_words = Counter(words)
    wordcloud = WordCloud(width = 1000, height = 500).generate_from_frequencies(counter_words)
    plt.imshow(wordcloud)
    plt.show()
    return counter_words.most_common(most_common_nb)


def reg_length(x):
    try:
        return re.sub(r'\b\w{1,3}\b', '', x)
    except:
        return x
    
    
def reg_spaces(x):
    try: 
        return re.sub(' +', ' ', x)
    except:
        return x
    
    
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
    
    

    
    
    
    
    
    
    
    
    
    
    