import requests
import pandas as pd
import json
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from youtube_transcript_api import YouTubeTranscriptApi

#import sys
#sys.path.append("app/algorithms")
import app.algorithms.process_predictions as process #Custom script to extract data

#Video ID to predict on
vid = "Gnhv3WFnmH8"

model = load_model("./data/models/nb_spot.h5")
with open("./data/spot_tokenizer_10k.json") as f:
    json_obj = json.load(f)
    tokenizer = tokenizer_from_json(json_obj)

transcript = YouTubeTranscriptApi.get_transcript(vid, languages = ["en"])
segments = requests.get("https://sponsor.ajay.app/api/skipSegments",params={"videoID":vid}).json()

text = []
for seg in segments:
    string = process.extractText(seg["segment"],transcript, widen = 0.05)[0]
    print("Predicting on <<" + string[0:100] + "...>>")
    text.append(string)

data = pd.DataFrame({"text":text})
x_new = tokenizer.texts_to_sequences(data["text"].values)
x_new = pad_sequences(x_new, padding = "post", maxlen = 3000, truncating = "post")

print(model.predict(x_new, batch_size = 1).round(3))
