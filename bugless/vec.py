import numpy as np
from scipy.sparse import csr_matrix, linalg, csgraph
from sklearn import metrics
import re
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import gudhi
import matplotlib.pyplot as plot
from nltk.tokenize import sent_tokenize

# Problems?: aimer -> aim in stemming, pas est un stopword
def preprocess(s, lang : str, remove_stop_words : bool):
  '''
  s being a corpus or a document
  return the cleaned token list(s)
  '''
  if(type(s) == str):
    tokens = re.sub(r'[^A-Za-zÀ-ÖØ-öø-ÿ]', ' ', s).lower().split()

    stemmer = SnowballStemmer(lang, ignore_stopwords=False)
    tokens = [stemmer.stem(token) for token in tokens]
    if remove_stop_words:
      stop = set(stopwords.words(lang))
      if lang == 'french':
        stop.remove('pas')
      elif lang == 'english':
        stop.remove('not')
      tokens = [token for token in tokens if token not in stop]

    if len(tokens) == 0:
      # print("What? ", s)
      tokens.append("e")
    return tokens
  else:
    res = []
    for t in s:
      t = preprocess(t, lang, remove_stop_words)
      if len(t) > 0:
        res.append(t)
    return res

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

def tfidf(docs : list[list[str]]):
  '''
  Take a list of lists of words, return a tuple of (the dictionary, idf, tfidf)
  '''
  tf = bag_of_words(docs)

  dc = np.zeros((tf.shape[1],), dtype=np.float64)
  for i in range(tf.shape[0]):
    dc[np.where(tf[i] > 0)] += 1
  idf = np.log(len(docs) / dc)

  for i, doc in enumerate(docs):
    tf[i] /= len(doc)

  tfidf = tf * idf
  return tfidf

def bag_of_words(docs : list[list[str]]):
  terms = []
  for doc in docs:
    terms.extend(doc)
  terms = np.unique(terms)

  lk = dict(zip(terms, range(len(terms))))
  tc = np.zeros((len(docs), len(terms)), dtype=np.float64)
  for i, doc in enumerate(docs):
    indices, counts = np.unique([lk[term] for term in doc], return_counts=True)
    if len(indices) > 0:
      tc[i, indices] = counts
  return tc

def barcodes_in_time_skeleton(vecs : np.ndarray, visual = False):
  distances = metrics.pairwise_distances(vecs, metric='cosine', n_jobs=-1)
  n = distances.shape[0]
  for i in range(n-1):
    distances[i+1, i] = distances[i, i+1] = 0
  rips_complex = gudhi.RipsComplex(distance_matrix = distances)
  simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
  diag = simplex_tree.persistence(homology_coeff_field=2, min_persistence=0)
  if visual:
    gudhi.plot_persistence_diagram(diag)
    plot.show()
  return diag

def cnt_barcodes(diag, d : int) -> int:
  return len([1 for p in diag if p[0] == d])

def essay_homology(essay : str, visual = False):
  corpus = sent_tokenize(essay)
  corpus = preprocess(corpus, 'english', True)
  vecs = tfidf(corpus)
  return barcodes_in_time_skeleton(vecs, visual)