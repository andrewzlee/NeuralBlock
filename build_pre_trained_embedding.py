#Instead of training our own embeddings we'll using the embeddings
#from Facebook's fastText project 
#https://fasttext.cc/docs/en/english-vectors.html

import numpy as np
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json
import pickle

#Code from fast text doc page.
import io
def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data


MAX_WORDS = 10000
EMBEDDING_DIM = 300 #word embedding length
embeddings_index = load_vectors("./data/embeddings/wiki-news-300d-1M.vec")

#Get Tokenizer from train_stream
with open("./data/tokenizer_stream_10k.json") as f:
    json_obj = json.load(f)
    tokenizer = tokenizer_from_json(json_obj)

word_index = tokenizer.word_index
num_words = min(MAX_WORDS,len(word_index) + 1)
                
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
cnt = 0
for word, i in word_index.items():
    if i >= num_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = list(embedding_vector)
    else:
        cnt+=1
        print(f"('{word}', {i}) was not found.")
print(cnt)
pickle.dump(embedding_matrix, open("./data/embedding_matrix_10k.pkl", "wb"))

