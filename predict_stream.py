from tensorflow.keras.models import load_model
import json
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from youtube_transcript_api import YouTubeTranscriptApi
import re
import pandas as pd
import app.algorithms.process_predictions as pp

MAXWORDS = 3000

model = load_model("./data/models/nb_stream_fasttext_10k.h5")
with open("./data/tokenizer_stream_10k.json") as f:
    json_obj = json.load(f)
    tokenizer = tokenizer_from_json(json_obj)

vid = "QqEuO5im1nA"
channel = "HalfAsInteresting" #For file naming only

transcript = YouTubeTranscriptApi.get_transcript(vid, languages = ["en"])

chars = "(!|\"|#|\$|%|&|\(|\)|\*|\+|,|-|\.|/|:|;\<|=|>|\?|@|\[|\\\\|\]|\^|_|`|\{|\||\}|~|\t|\n)+"
captionCount = []
full_text = ""
for t in transcript:
    cleaned_text = re.sub("  +", " ", re.sub(chars, " ", t["text"])).strip()
    captionCount.append(len(cleaned_text.split(" ")))
    full_text = full_text + " " + cleaned_text
full_text = full_text.strip()

predictions, status = pp.getPredictions(model,tokenizer,full_text)

if status:
    df = pd.DataFrame(predictions[0,:,:])
    words = full_text.split(" ")
    df["text"] =  words + ["N/A"]*(MAXWORDS-len(words))
    df.to_csv(f"./examples/{channel}_{vid}.csv", index = False)

    sponsorTimestamps,sponsorText = pp.getTimestamps(transcript, captionCount, predictions[0], words)
    print(sponsorTimestamps)     
   
    with open(f"./examples/{channel}_{vid}.txt", 'w') as file:
        file.write("Timestamps:\n")
        for ts in sponsorTimestamps:
            file.write('%s\n' % str(ts))
else:
    print("Failed")