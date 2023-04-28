import numpy as np
from scipy.sparse import csr_matrix, linalg, csgraph
from sklearn import metrics
import re
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords

# Problems?: aimer -> aim in stemming
def preprocess(s, lang, remove_stop_words):
  '''
  s being a corpus or a document
  return the cleaned token list(s)
  '''
  if(type(s) == str):
    tokens = re.sub(r'[^\w]', ' ', s).lower().split()

    stemmer = SnowballStemmer(lang, ignore_stopwords=False)
    tokens = [stemmer.stem(token) for token in tokens]
    if remove_stop_words:
      stop = set(stopwords.words(lang))
      tokens = [token for token in tokens if token not in stop]

    return tokens
  else:
     return [preprocess(t, lang, remove_stop_words) for t in s]

# O(n^2 m)
def compute_barcodes(X : csr_matrix, metric) -> np.ndarray:
  '''
  metric in 'cosine', 'euclidean', 'l1', 'l2'
  '''
  n = X.shape[0]
  distances = metrics.pairwise_distances(X, metric='euclidean', n_jobs=-1)
  deathes = [np.min([distances[i, j] for j in range(i+1, n)], initial=np.inf) for i in range(n)]
  return sorted(deathes)

def connected_components_under_dist(X : csr_matrix, dist_lim : float, metric):
  '''
  metric in 'cosine', 'euclidean', 'l1', 'l2'
  '''
  G = csr_matrix(metrics.pairwise_distances(X, metric=metric, n_jobs=-1)<dist_lim)
  return csgraph.connected_components(G)

# O(nm log(nm))
def tfidf(docs : list[list[str]]) -> tuple[list[str], np.ndarray, list[np.ndarray]]:
  '''
  Take a list of lists of words, return a tuple of (the dictionary, idf, tfidf)
  '''
  words = []
  for doc in docs:
      words.extend(doc)
  words = np.unique(words)

  lk = dict(zip(words, range(len(words)))) # word to id
  tf = np.zeros((len(docs), len(words)), dtype=np.float64)
  for i, doc in enumerate(docs):
      indices, counts = np.unique([lk[term] for term in doc], return_index=True)
      tf[i, indices] = counts/len(doc)

  dc = np.zeros((len(words),), dtype=np.float64)
  for doc in docs:
      dc[np.unique([lk[term] for term in doc])] += 1
  idf = np.log(len(docs) / dc)

  tfidf = tf * idf
  return words, idf, tfidf