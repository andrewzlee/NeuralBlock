import pandas as pd
from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer
import json
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, SpatialDropout1D
from keras.preprocessing.sequence import pad_sequences

data = pd.read_pickle("./data/data.pkl")

max_features = 2000
tokenizer = Tokenizer(num_words = max_features)
tokenizer.fit_on_texts(data["text"].values)

with open("./data/tokenizer.json", "w", encoding ="utf-8") as f:
    jsonobj = tokenizer.to_json()
    f.write(json.dumps(jsonobj, ensure_ascii = False))


X = tokenizer.texts_to_sequences(data["text"].values)
X = pad_sequences(X)
Y = pd.get_dummies(data["sponsor"]).values

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 2020)

embed_dim = 128
lstm_out = 196

model = Sequential()
model.add(Embedding(max_features, embed_dim, input_length = X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())

batch_size = 32
model.fit(x_train, y_train, validation_split = 0.1, epochs = 7, batch_size=batch_size, verbose = 1)#, use_multiprocessing = True)
model.save("./data/nb_model.h5")

score,acc = model.evaluate(x_test, y_test, verbose = 1, batch_size = batch_size)

