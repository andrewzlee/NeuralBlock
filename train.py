import pandas as pd
from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer
import json
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, SpatialDropout1D
from keras.preprocessing.sequence import pad_sequences

data = pd.read_pickle("./data/data.pkl")

maxwords = 10000
tokenizer = Tokenizer(num_words = maxwords, oov_token = "oovword")
tokenizer.fit_on_texts(data["text"].values)

#Write tokenizer to json for future reference.
with open("./data/spot_tokenizer.json", "w", encoding ="utf-8") as f:
    jsonobj = tokenizer.to_json()
    f.write(json.dumps(jsonobj, ensure_ascii = False))


X = tokenizer.texts_to_sequences(data["text"].values)
X = pad_sequences(X, padding = "post") #maxlen = 3000, truncating = "post"
Y = pd.get_dummies(data["sponsor"]).values #Convert to two columns

del X, Y

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 2020)

embedding_matrix = pickle.load(open("./data/embedding_matrix_10k.pkl","rb"))

embed_dim = 300
lstm_out = 196

maxseqlen = x_train.shape[1]
eb_layer = Embedding(maxwords, embed_dim, embeddings_initializer=Constant(embedding_matrix), 
                     input_length=maxseqlen, mask_zero=True, trainable=False)

model = Sequential()
model.add(eb_layer)
model.add(SpatialDropout1D(0.4))
model.add(Bidirectional(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2)))
model.add(Dense(2,activation='softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())

batch_size = 32
model.fit(x_train, y_train, validation_split = 0.1, epochs = 7, batch_size=batch_size, verbose = 1)#, use_multiprocessing = True)
model.save("./data/spot_model.h5")

score,acc = model.evaluate(x_test, y_test, verbose = 1, batch_size = batch_size)

