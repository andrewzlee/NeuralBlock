#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import re
from youtube_transcript_api import YouTubeTranscriptApi

#Use the predictions and work backwards from the transcripts to get the 
#approximate timestamps for a sponsored segment.

vid = "5hjqXaez7ac"
f = "examples/SciShow_5hjqXaez7ac.csv"

df = pd.read_csv(f)
transcript = YouTubeTranscriptApi.get_transcript(vid, languages = ["en"])

sponsorSegments = []
startIdx = 0
sFlag = 0
thresh = 0.55
for index,row in df.iterrows():
    if row["1"] >= thresh and not sFlag:
        startIdx = index
        sFlag = 1
    if row["1"] < thresh and sFlag:
        sponsorSegments.append((startIdx,index))
        sFlag = 0

chars = "(!|\"|#|\$|%|&|\(|\)|\*|\+|,|-|\.|/|:|;\<|=|>|\?|@|\[|\\\\|\]|\^|_|`|\{|\||\}|~|\t|\n)+"
captionCount = []
for t in transcript:
    cleaned_text = re.sub("  +", " ", re.sub(chars, " ", t["text"])).strip()
    captionCount.append(len(cleaned_text.split(" ")))

sponsorTimestamps = []
sFlag = 0
wpm = 3
for segs in sponsorSegments:
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
    sponsorTimestamps.append((startTime,endTime))

print(sponsorTimestamps)     
   
with open(f[:-4]+".txt", 'w') as file:
    file.write("Timestamps:\n")
    for ts in sponsorTimestamps:
        file.write('%s\n' % str(ts))