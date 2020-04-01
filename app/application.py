#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Flask, render_template, request

from tensorflow.keras.models import load_model
import json
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from youtube_transcript_api import YouTubeTranscriptApi
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

app = Flask(__name__)

model = load_model("./models/nb_stream_fasttext_10k.h5")
with open("./models/tokenizer_stream_10k.json") as f:
    json_obj = json.load(f)
    tokenizer = tokenizer_from_json(json_obj)


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST","GET"])
def predict():
    vid = request.form["vid"]
    transcript = YouTubeTranscriptApi.get_transcript(vid, languages = ["en"])
    
    chars = "(!|\"|#|\$|%|&|\(|\)|\*|\+|,|-|\.|/|:|;\<|=|>|\?|@|\[|\\\\|\]|\^|_|`|\{|\||\}|~|\t|\n)+"
    captionCount = []
    full_text = ""
    for t in transcript:
        cleaned_text = re.sub("  +", " ", re.sub(chars, " ", t["text"])).strip()
        captionCount.append(len(cleaned_text.split(" ")))
        full_text = full_text + " " + cleaned_text
    full_text = full_text.strip()
    predictions = getPredictions(full_text)
    sponsorTimestamps,sponsorText = getTimestamps(transcript, captionCount, predictions[0], full_text.split(" "))
    minuteStamps = []
    for t in sponsorTimestamps:
        m1,s1 = divmod(round(t[0]),60)
        m2,s2 = divmod(round(t[1]),60)
        minuteStamps.append(f"({m1}:{str(s1).zfill(2)},{m2}:{str(s2).zfill(2)})")
    
    return render_template("predict.html", videoid = vid, transcript_text = full_text, 
                           timestamp = " ".join(str(e) for e in sponsorTimestamps),
                           minutestamp = " ".join(minuteStamps),
                           sponsTexts = sponsorText)

def getPredictions(text):
    full_seq = tokenizer.texts_to_sequences([text])
    seqlen = len(full_seq[0])
    print("Sequence length: {}".format(seqlen))
    #Videos with more than 3000 words need to be split up.
    maxseqlen = 3000
    if seqlen <= maxseqlen:
        full_seq = pad_sequences(full_seq, maxlen = maxseqlen, padding = "post")
        return model.predict(full_seq, batch_size = 1).round(3)
    else:
        print("Video needs to be 3000 words or fewer.")
    return []

def getTimestamps(transcript, captionCount, predictions, words):
    sponsorSegments = []
    startIdx = 0
    #Minimum confidence to start Sponsor
    thresh = 0.60
    sFlag = 0
    #Sponsor confidence must exceed this value at some pont
    confidence = 0.80 
    sFlagConf = 0 

    for index,row in enumerate(predictions):
        if row[1] >= thresh and not sFlag:
            startIdx = index
            sFlag = 1
            continue

        if sFlag and row[1] >= confidence:
            sFlagConf = 1

        if row[1] < thresh and sFlag:
            if sFlagConf and (index-startIdx)>5: #Exclude segments with <= 5 words.
                sponsorSegments.append((startIdx,index))
            else:
                print(f"The segment ({startIdx},{index}) was tossed.")
            #Reset flags
            sFlag = 0
            sFlagConf = 0
            
    sponsorTimestamps = []
    sponsorText = []
    sFlag = 0
    wpm = 3
    for segs in sponsorSegments:
        sponsorText.append(" ".join(words[segs[0]:segs[1]]))
        numWords = 0
        for idx, e in enumerate(captionCount):
            if numWords <= segs[0] <= numWords+e and not sFlag:
                excessHead = max(segs[0] - numWords - 1, 0) #every word before the starting word
                startIdx = idx
                sFlag = 1
            
            numWords += e
            
            if numWords >= segs[1] and sFlag:
                excessTail = segs[1]-(numWords-e) #how many words in the next caption to keep
                endIdx = idx
                sFlag = 0
                break
        
        startTime = transcript[startIdx]["start"] + excessHead/wpm
        endTime = min(transcript[endIdx]["start"] + transcript[endIdx]["duration"], 
                      transcript[endIdx]["start"] + excessTail/wpm)
        sponsorTimestamps.append((round(startTime,3),round(endTime,3)))
    return sponsorTimestamps, sponsorText

if __name__ == "__main__":
    app.run(debug=True)