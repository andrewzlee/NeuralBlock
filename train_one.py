import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, SpatialDropout1D, Bidirectional
from tensorflow.keras.initializers import Constant
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# import tensorflow as tf
# gpu_devices = tf.config.experimental.list_physical_devices('GPU')
# for device in gpu_devices: tf.config.experimental.set_memory_growth(device, True)

data = pd.read_pickle("./data/spot_data.pkl")

#Get Tokenizer created in prepare_spot.py
with open("./data/spot_tokenizer_10k.json") as f:
    json_obj = json.load(f)
    tokenizer = tokenizer_from_json(json_obj)


X = tokenizer.texts_to_sequences(data["text"].values)
X = pad_sequences(X, padding = "post", maxlen = 3000, truncating = "post")
Y = pd.get_dummies(data["sponsor"]).values #Convert to two columns

#Delete completely masked
masked_rows = np.where(~X.any(axis=1))[0] 
X = np.delete(X, masked_rows , axis = 0)
Y = np.delete(Y, masked_rows , axis = 0)


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 20200629) #Date in YYYYMMDD

del X, Y, tokenizer, json_obj, data, f, masked_rows

embedding_matrix = pickle.load(open("./data/embedding_matrix_10k.pkl","rb"))

embed_dim = 300
lstm_out = 128
maxwords = 10000
batch_size = 22

maxseqlen = x_train.shape[1]
eb_layer = Embedding(maxwords, embed_dim, embeddings_initializer=Constant(embedding_matrix), 
                     input_length=maxseqlen, mask_zero=True, trainable=False)

model = Sequential()
model.add(eb_layer)
model.add(SpatialDropout1D(0.3))
model.add(Bidirectional(LSTM(lstm_out, dropout=0.2, batch_input_shape = (batch_size, x_train.shape[0], maxseqlen))))
model.add(Dense(2,activation='softmax'))

model.compile(loss = "categorical_crossentropy", optimizer="adam",metrics = ["accuracy"])
print(model.summary())

model.fit(x_train, y_train, epochs = 15, batch_size=batch_size, verbose = 1)
model.save("./data/models/nb_spot_10k.h5")

score,acc = model.evaluate(x_test, y_test, verbose = 1, batch_size = batch_size)

