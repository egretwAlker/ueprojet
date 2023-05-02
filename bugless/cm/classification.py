from tqdm import tqdm
tqdm.pandas()
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder

def naive_bayes_train(X, y, parameters_pos, parameters_neg): 
    
    cv = CountVectorizer(max_features=30000)
    X_train_array = cv.fit_transform(X)
    X_train_array = X_train_array.toarray()
    
    vocab = cv.get_feature_names_out().tolist()
    p_pos, p_neg = y_train.value_counts(normalize=True)[0], y_train.value_counts(normalize=True)[1]
    alpha = 1
    
    n_pos = X_train.loc[y_train == 1].apply(len).sum()
    n_neg = X_train.loc[y_train == 0].apply(len).sum()
    n_vocab = len(vocab)
    
    parameters_pos = {word:0 for word in vocab}
    parameters_neg = {word:0 for word in vocab}
    
    for word in tqdm(vocab):
        n_word_given_pos = vocab_counts.loc[y_train == True, word].sum()
        n_word_given_neg = vocab_counts.loc[y_train == False, word].sum()
    
    parameters_pos[word] = (n_word_given_pos + alpha) / (n_pos + alpha * n_vocab)
    parameters_neg[word] = (n_word_given_neg + alpha) / (n_neg + alpha * n_vocab)
    
    return p_pos, p_neg, parameters_pos, parameters_neg
    
def classify_review(x, p_pos, p_neg, parameters_pos, parameters_neg):
    p_pos_given = p_pos
    p_neg_given = p_neg
    
    for word in x:
        if word in parameters_pos:
            p_pos_given *= parameters_pos[word]
            
        if word in parameters_neg:
            p_neg_given *= parameters_neg[word]
            
    if p_pos_given > p_neg_given:
        return 1
    else: 
        return 0