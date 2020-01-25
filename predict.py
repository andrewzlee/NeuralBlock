import pandas as pd
import json
from keras.preprocessing.text import tokenizer_from_json
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from youtube_transcript_api import YouTubeTranscriptApi

model = load_model("./data/nb_model.h5")
with open("./data/tokenizer.json") as f:
    json_obj = json.load(f)
    tokenizer = tokenizer_from_json(json_obj)

vid = "N1rTDbjc_fI"

transcript = YouTubeTranscriptApi.get_transcript(vid, languages = ["en"])

text1 = ""
for i in range(0,3):
    text1 = text1 + " " + transcript[i]["text"]
    
text2 = ""
for i in range(24,39):
    text2 = text2 + " " + transcript[i]["text"]
    
text3 = ""
for i in range(76,88):
    text3 = text3 + " " + transcript[i]["text"]

dfnew = pd.DataFrame({"text":[text1,text2,text3]})
x_new = tokenizer.texts_to_sequences(dfnew["text"].values)
x_new = pad_sequences(x_new, maxlen=3943)
print(model.predict(x_new, batch_size = 1).round(3))