

def count_vector_encode(com_words, sentence, count_matrix, row_index):
    for word in sentence.split(' '):
        if word in com_words:
            count_matrix[row_index, com_words.index(word)] += 1