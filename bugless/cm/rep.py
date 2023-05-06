

def count_vector_encode(com_words, sentence, count_matrix, row_index):
    for word in sentence.split(' '):
        if word in com_words:
            count_matrix[row_index, com_words.index(word)] += 1
            
# gets list of words as argument
def one_hot_encoding(sentences):
    token_index = {}
    #Create a counter for counting the number of key-value pairs in the token_length
    counter = 0

    # Select the elements of the samples which are the two sentences
    for sentence in sentences:   
        for considered_word in sentence.split():
        if considered_word not in token_index:

            # If the considered word is not present in the dictionary token_index, add it to the token_index
            # The index of the word in the dictionary begins from 1 
            token_index.update({considered_word : counter + 1}) 

            # updating the value of counter
            counter = counter + 1                        
    
    max_length = 50
    # Create a tensor of dimension 3 named results whose every elements are initialized to 0
    results  = np.zeros(shape = (len(words),
                            max_length,
                            max(token_index.values()) + 1))  
    
    for i, sentence in enumerate(sentences): 
      # Convert enumerate object to list and iterate over resultant list 
      for j, considered_word in list(enumerate(sentence.split())):
    
        # set the value of index variable equal to the value of considered_word in token_index
        index = token_index.get(considered_word)
    
        # In the previous zero tensor: results, set the value of elements with their positional index as [i, j, index] = 1.
        results[i, j, index] = 1.