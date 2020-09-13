#Using a stream of text, and a word embedding matrix provided by fastText
#we build a RNN that produces P(Sponsor|Word) for all words in the sequence.
#
#Training takes ~3 hr/epoch on a Nvidia RTX 2080ti
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, SpatialDropout1D, TimeDistributed, Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import optimizers 
#from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.initializers import Constant

X_cleaned = pickle.load(open("./data/x_stream_10k.pkl", "rb"))
X_cleaned = pad_sequences(X_cleaned, padding = "post")
Y_cleaned = pickle.load(open("./data/y_stream_cat_10k.pkl","rb"))
sample_weights = pickle.load(open("./data/sample_weights_10k.pkl", "rb"))

x_train, x_test, y_train, y_test, sw_train, sw_test = train_test_split(X_cleaned, Y_cleaned, sample_weights, test_size = 0.25, random_state = 2020)

del X_cleaned, Y_cleaned, sample_weights

embedding_matrix = pickle.load(open("./data/embedding_matrix_10k.pkl","rb"))

embed_dim = 300
lstm_out = 128
maxwords = 10000
maxseqlen = x_train.shape[1]

eb_layer = Embedding(maxwords, embed_dim, embeddings_initializer=Constant(embedding_matrix), 
                     input_length=maxseqlen, mask_zero=True, trainable=False)
model = Sequential()
model.add(eb_layer)
model.add(SpatialDropout1D(0.30))
model.add(Bidirectional(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))
model.add(TimeDistributed(Dense(2,activation="softmax")))

adam = optimizers.Adam(learning_rate = 0.001)
model.compile(loss="binary_crossentropy", optimizer=adam, metrics=["acc"], sample_weight_mode="temporal")
print(model.summary())

batch_size = 32
#class_weights = {0: 1., 1: 5.}
weighting = 4
model.fit(x_train, y_train, validation_split = 0.1, epochs = 10, sample_weight = sw_train*weighting+1,
          batch_size=batch_size, verbose = 1)

model.save("./data/models/nb_stream_10_fasttext.h5")

score,acc = model.evaluate(x_test, y_test, verbose = 1, batch_size = batch_size)