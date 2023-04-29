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
    tokens = re.sub(r'[^\w]|[\d]', ' ', s).lower().split()

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
def tfidf(docs : list[list[str]]) -> np.ndarray:
  '''
  Take a list of lists of words, return a tuple of (the dictionary, idf, tfidf)
  '''
  tf = bag_of_words(docs)

  dc = np.zeros((tf.shape[1],), dtype=np.float64)
  for i in range(tf.shape[0]):
    dc[np.where(tf[i] > 0)] += 1
  idf = np.log(len(docs) / dc)

  for i in range(dc.shape[0]):
    if dc[i] == 0.0:
      print(i)

  for i, doc in enumerate(docs):
    tf[i] /= len(doc)

  tfidf = tf * idf
  return tfidf

def bag_of_words(docs : list[list[str]]) -> np.ndarray:
  terms = []
  for doc in docs:
    terms.extend(doc)
  terms = np.unique(terms)

  lk = dict(zip(terms, range(len(terms)))) # word to id
  tc = np.zeros((len(docs), len(terms)), dtype=np.float64)
  for i, doc in enumerate(docs):
    indices, counts = np.unique([lk[term] for term in doc], return_counts=True)
    tc[i, indices] = counts
    # if i == 5369:
    #   print(indices, counts)
    #   print(tc[i, 123])
  return tc