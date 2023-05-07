import numpy as np
import random
from scipy.sparse import csr_matrix, vstack
def finalexam(labels : list[bool], vectors : csr_matrix, model, split_rate = 0.8, random_seed = 233):
  """
  The model takes m labels, m vectors (for training), n-m vectors (to predict)
  For now, labels are true/false
  """
  random.seed(random_seed) # to ensure reproductivity
  train_labels, train_vectors, test_labels, test_vectors = [], [], [], []
  n = len(labels)
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
  return {
    "Guess false to false": cnt[0][0],
    "Guess false to true": cnt[0][1],
    "Guess true to false": cnt[1][0],
    "Guess true to true": cnt[1][1],
    "Accuracy when guessing false": cnt[0][0]/(cnt[0][0]+cnt[0][1]),
    "Accuracy when guessing true": cnt[1][1]/(cnt[1][0]+cnt[1][1]),
    "Accuracy in total": (cnt[0][0]+cnt[1][1])/len(output)
  }