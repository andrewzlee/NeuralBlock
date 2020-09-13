import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import json
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#keras.utils.to_categorical(y, num_classes=None, dtype='float32')
def to_categorical(sequences, categories):
    cat_sequences = []
    for s in sequences:
        cats = []
        for item in s:
            cats.append(np.zeros(categories))
            cats[-1][item] = 1.0
        cat_sequences.append(cats)
    return np.array(cat_sequences)

def trimData(X, Y, max_words = 4000, overlap = 1000):
    #Clip videos to be a max of 4,000 words.
    X_cleaned = []
    Y_cleaned = []
    
    for i in range(len(X)):
        text = X[i]
        numwords = len(text)
        labels = Y[i]
        
        assert numwords == len(labels)
        
        if numwords<5:
            print('Skipping corrupt data')
            continue
        
        #Cap at about 3.3 words/sec or 4000 words per 20 minutes
        if numwords > max_words:
            i = 0
            while True:
                if ((max_words-overlap)*i+max_words) > numwords:
                    break
                elif i == 0: #first case
                    X_cleaned.append(  text[0:max_words])
                    Y_cleaned.append(labels[0:max_words])
                else: 
                    #overlapping 5 minutes
                    X_cleaned.append(  text[(max_words-overlap)*i:((max_words-overlap)*i+max_words)])
                    Y_cleaned.append(labels[(max_words-overlap)*i:((max_words-overlap)*i+max_words)])
                i += 1
                
            #last chunk
            X_cleaned.append(  text[(max_words-overlap)*i:numwords])    
            Y_cleaned.append(labels[(max_words-overlap)*i:numwords])
        else:
            X_cleaned.append(text)
            Y_cleaned.append(labels)
    return X_cleaned, Y_cleaned


maxwords = 10000
engine = create_engine(r"sqlite:///data/labeled.db")
#Build a dictionary on only words used in sponsorships
sponsor_vocabulary = pd.read_sql("select videoid, text from sponsordata where processed = 1",engine)
tokenizer = Tokenizer(num_words = maxwords, oov_token = "oovword")
tokenizer.fit_on_texts(sponsor_vocabulary["text"].values)

streamdata = pd.read_sql("select * from sponsorstream where length(text) >= 10", engine)

X = tokenizer.texts_to_sequences(streamdata["text"].values)
Y = []
#Convert the labels into integers
for row in streamdata.itertuples(index = False):
    Y.append([int(s) for s in row[2][1:-1].split(', ')])

X_cleaned, Y_cleaned = trimData(X,Y,max_words = 3000, overlap = 800)

sample_weights = pad_sequences(Y_cleaned, padding = "post")
pickle.dump(sample_weights, open("./data/sample_weights_10k.pkl", "wb"))
del sample_weights

pickle.dump(X_cleaned, open("./data/x_stream_10k.pkl", "wb"))
del X_cleaned

Y_cleaned = to_categorical(Y_cleaned, 2)
Y_cleaned = pad_sequences(Y_cleaned, padding = "post")
pickle.dump(Y_cleaned, open("./data/y_stream_cat_10k.pkl","wb"))

with open("./data/tokenizer_stream_10k.json", "w", encoding ="utf-8") as f:
    jsonobj = tokenizer.to_json()
    f.write(json.dumps(jsonobj, ensure_ascii = False))

