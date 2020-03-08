from tensorflow.keras.models import load_model
import json
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from youtube_transcript_api import YouTubeTranscriptApi
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model("./data/models/nb_stream_5.h5")
with open("./data/tokenizer_stream_10k.json") as f:
    json_obj = json.load(f)
    tokenizer = tokenizer_from_json(json_obj)

vid = "6hJ_E2NOdcw"

transcript = YouTubeTranscriptApi.get_transcript(vid, languages = ["en"])

full_text = ""
for i in transcript:
    full_text = full_text + " " + i["text"]

full_text = full_text.replace("\n", " ")
print(full_text)
full_seq = tokenizer.texts_to_sequences([full_text])
print(len(full_seq[0]))
full_seq = pad_sequences(full_seq, maxlen = 3000, padding = "post")
results = model.predict(full_seq, batch_size = 1).round(3)