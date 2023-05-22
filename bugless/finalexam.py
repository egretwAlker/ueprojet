import numpy as np
import random
from scipy.sparse import csr_matrix, vstack
from numpy.typing import ArrayLike
def finalexam(labels : ArrayLike, vectors : csr_matrix, model, split_rate = 0.8, random_seed = 233):
  """
  labels are bools
  The model takes m labels, m vectors (for training), n-m vectors (to predict)
  For now, labels are true/false
  """
  labels = np.array(labels)
  random.seed(random_seed) # to ensure reproductivity
  train_labels, train_vectors, test_labels, test_vectors = [], [], [], []
  n = labels.shape[0]
  for i in range(n):
    if random.random() < split_rate:
      train_labels.append(labels[i])
      train_vectors.append(vectors[i])
    else:
      test_labels.append(labels[i])
      test_vectors.append(vectors[i])
  train_vectors = vstack(train_vectors)
  test_vectors = vstack(test_vectors)
  output = list(model(train_labels, train_vectors, test_vectors))
  cnt = [[0, 0], [0, 0]]
  for i in range(len(output)):
    # print(int(test_labels[i]), output[i])
    cnt[int(test_labels[i])][int(output[i])] += 1
  p = cnt[0][0]/(cnt[0][0]+cnt[0][1])
  q = cnt[1][1]/(cnt[1][0]+cnt[1][1])
  return {
    "True negatives": cnt[0][0],
    "False positives": cnt[0][1],
    "False negatives": cnt[1][0],
    "True positive": cnt[1][1],
    "Accuracy on falses": p,
    "Accuracy on trues": q,
    "Accuracy in total": (cnt[0][0]+cnt[1][1])/len(output),
    "Accuracy in average": (p+q)*0.5
  }