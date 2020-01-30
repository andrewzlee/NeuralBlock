#import pandas as pd
from keras.models import load_model
import json
from keras.preprocessing.text import tokenizer_from_json
from youtube_transcript_api import YouTubeTranscriptApi
from keras.preprocessing.sequence import pad_sequences
import numpy as np

model = load_model("./data/nb_model_stream.h5")
with open("./data/tokenizer_stream.json") as f:
    json_obj = json.load(f)
    tokenizer = tokenizer_from_json(json_obj)

vid = "vpcUVOjUrKk"

transcript = YouTubeTranscriptApi.get_transcript(vid, languages = ["en"])

text1 = ""
for i in range(0,3):
    text1 = text1 + " " + transcript[i]["text"]
text1 = text1.replace("\n"," ")
t1 = tokenizer.texts_to_sequences([text1])
l1 = [1] * len(t1[0])

text2 = ""
for i in range(3,187):
    text2 = text2 + " " + transcript[i]["text"]
text2 = text2.replace("\n"," ")
t2 = tokenizer.texts_to_sequences([text2])
l2 = [0] * len(t2[0])

text3 = ""
for i in range(187,206):
    text3 = text3 + " " + transcript[i]["text"]
text3 = text3.replace("\n"," ")
t3 = tokenizer.texts_to_sequences([text3])
l3 = [1] * len(t3[0])

t_final = t1[0] + t2[0] + t3[0]
l_final = l1 + l2 + l3

t_final = pad_sequences([t_final], maxlen = 4000)
l_final = np.transpose(pad_sequences([l_final], maxlen = 4000))
results = model.predict(t_final, batch_size = 1).round(3)