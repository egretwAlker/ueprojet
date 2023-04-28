import numpy as np
from scipy.sparse import csr_matrix, linalg, csgraph
from sklearn import metrics
import re

# O(n^2 m)
def compute_barcodes(X : csr_matrix) -> np.ndarray:
    n = X.shape[0]
    distances = metrics.pairwise_distances(X, metric='euclidean', n_jobs=-1)
    deathes = [np.min([distances[i, j] for j in range(i+1, n)], initial=np.inf) for i in range(n)]
    return sorted(deathes)

def connected_components_under_dist(X : csr_matrix, dist_lim : float):
  n = X.shape[0]
  G = csr_matrix(metrics.pairwise_distances(X, metric='euclidean', n_jobs=-1)<dist_lim)
  return csgraph.connected_components(G)

def my_split(docs : list[str]) -> list[list[str]]:
  return [[w.lower() for w in re.sub(r'[^\w]', ' ', s).split() if w.isalpha() and len(w) >= 3] for s in docs]

# O(nm log(nm))
def tfidf(docs : list[list[str]]) -> tuple[list[str], np.ndarray, list[np.ndarray]]:
  '''
  Take a list of list of words, return a tuple of (the dictionary, idf, tfidf)
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