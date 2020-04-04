from tensorflow.keras.models import load_model
import json
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import pandas as pd
import app.algorithms.process_predictions as pp

MAXWORDS = 3000

model = load_model("./data/models/nb_stream_fasttext_10k.h5")
with open("./data/tokenizer_stream_10k.json") as f:
    json_obj = json.load(f)
    tokenizer = tokenizer_from_json(json_obj)

vid = "cnpUNEWP1i8"
channel = "MentourPilot" #For file naming only

transcript, full_text, captionCount = pp.processVideo(vid)
predictions = pp.getPredictions(model,tokenizer,full_text)

df = pd.DataFrame(predictions)
words = full_text.split(" ")
df["text"] =  words + ["N/A"]*(len(predictions)-len(words))
df.to_csv(f"./examples/{channel}_{vid}.csv", index = False)

sponsorTimestamps = pp.getTimestamps(transcript, captionCount, predictions, words)
print(sponsorTimestamps)     
   
with open(f"./examples/{channel}_{vid}.txt", 'w') as file:
    file.write("Timestamps:\n")
    for ts in sponsorTimestamps:
        file.write('%s\n' % str(ts))
