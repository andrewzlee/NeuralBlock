import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import requests
import pandas as pd
import json
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from youtube_transcript_api import YouTubeTranscriptApi
import random

#import sys
#sys.path.append("app/algorithms")
import app.algorithms.process_predictions as process #Custom script to extract data

#Video ID to predict on
vid = "46gNvDLgLdI"

model = load_model("./data/models/nb_spot.h5")
with open("./data/spot_tokenizer_10k.json") as f:
    json_obj = json.load(f)
    tokenizer = tokenizer_from_json(json_obj)

transcript = YouTubeTranscriptApi.get_transcript(vid, languages = ["en"])
segments = requests.get("https://sponsor.ajay.app/api/skipSegments",params={"videoID":vid}).json()

#Sponsored Segments
text = []
for seg in segments:
    string = process.extractText(seg["segment"],transcript, widen = 0.05)[0]
    print("[SPONSOR]:: <<" + string + ">>")
    text.append(string)

#Random 10 second segment
retry = 1
counter = 0
while retry:
    start_point = random.sample(transcript,1)[0]["start"]
    retry = 0
    counter += 1
    #Quit after 20 tries
    if counter >= 20:
        print("[NON-SPONSOR]:: NONE FOUND")
        break

    for seg in segments:
        #Check to ensure it's not in a sponsorship
        if seg["segment"][0] <= start_point <= seg["segment"][1]:
            retry = 1

if counter < 20:
    #Extract text
    string = process.extractText((start_point, start_point+10),transcript, widen = 0.05)[0]
    print("[NON-SPONSOR]:: <<" + string + ">>")
    text.append(string)

data = pd.DataFrame({"text":text})
x_new = tokenizer.texts_to_sequences(data["text"].values)
x_new = pad_sequences(x_new, padding = "post", maxlen = 3000, truncating = "post")

print(model.predict(x_new, batch_size = 1).round(3))
