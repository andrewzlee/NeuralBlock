from tensorflow.keras.models import load_model
import json
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from youtube_transcript_api import YouTubeTranscriptApi
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd

model = load_model("./data/models/nb_stream_fasttext_10k.h5")
with open("./data/tokenizer_stream_10k.json") as f:
    json_obj = json.load(f)
    tokenizer = tokenizer_from_json(json_obj)

vid = "aFyrgaC8iB4"

transcript = YouTubeTranscriptApi.get_transcript(vid, languages = ["en"])

full_text = ""
for i in transcript:
    full_text = full_text + " " + i["text"]

full_text = full_text.replace("\n", " ")
full_seq = tokenizer.texts_to_sequences([full_text])
seqlen = len(full_seq[0])
print("Sequence length: {}".format(seqlen))
#Videos with more than 3000 words need to be split up.
maxseqlen = 3000
if seqlen <= maxseqlen:
    print(full_text)
    full_seq = pad_sequences(full_seq, maxlen = 3000, padding = "post")
    full_text = tokenizer.sequences_to_texts(full_seq)[0].split(" ")
    results = model.predict(full_seq, batch_size = 1).round(3)
    df = pd.DataFrame(results[0,:,:])
    df["text"] = full_text
    df.to_csv(f"./examples/{vid}.csv", index = False)
else:
    print("Video needs to be 3000 words or fewer.")