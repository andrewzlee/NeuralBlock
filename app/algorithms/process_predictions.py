from tensorflow.keras.preprocessing.sequence import pad_sequences
from youtube_transcript_api import YouTubeTranscriptApi
import re
import numpy as np
import ds_stt as stt

def processVideo(vid, useDS = False):
    
    #Use DeepSpeech or YoutubeTranscriptApi to get transcript
    if useDS:
        stt.yt_download(vid)
        transcript, fullText = stt.transcribe(vid + ".wav")
        captionCount = [1] * len(transcript)
        
    else:
        transcript = YouTubeTranscriptApi.get_transcript(vid, languages = ["en"])

        chars = "(!|\"|#|\$|%|&|\(|\)|\*|\+|,|-|\.|/|:|;\<|=|>|\?|@|\[|\\\\|\]|\^|_|`|\{|\||\}|~|\t|\n)+"
        captionCount = []
        fullText = ""
        for t in transcript:
            cleaned_text = re.sub("  +", " ", re.sub(chars, " ", t["text"])).strip()
            captionCount.append(len(cleaned_text.split(" "))) #num words in the caption
            fullText = fullText + " " + cleaned_text
        fullText = fullText.strip()

    return transcript, fullText, captionCount

def getPredictions(model,tokenizer,text):
    full_seq = tokenizer.texts_to_sequences([text])
    numWords = len(full_seq[0])
    print("Sequence length: {}".format(numWords))
    #Videos with more than 3000 words need to be split up.
    maxNumWords = 3000
    if numWords <= maxNumWords:
        full_seq = pad_sequences(full_seq, maxlen = maxNumWords, padding = "post")
        return model.predict(full_seq, batch_size = 1).round(3)[0]
    else:
        #Long videos will be split up with a small overlap
        overlap = 500
        full_seq = splitSeq(full_seq[0], numWords, maxNumWords, overlap)
        full_seq = pad_sequences(full_seq, maxlen = maxNumWords, padding = "post")
        prediction = model.predict(full_seq, batch_size = len(full_seq)).round(3)

        #Stitch the split predictions back together
        full_prediction = np.empty([0,2], dtype = np.float32)
        overlapTail = np.empty([0,2], dtype = np.float32)

        #Iterate through the splits
        for i in prediction:
            overlapRegion = np.empty([0,2], dtype = np.float32) 
            #First n words in the split, which overlaps with the last n words of the previous split
            overlapHead = i[0:overlap] 

            for j in range(len(overlapTail)): #First split is skipped because len = 0
                maxValue = max(overlapHead[j][1],overlapTail[j][1]) #Should be the same numbers, but grabbing max just in case
                np.append(overlapRegion,(1-maxValue, maxValue))

            full_prediction = np.concatenate((full_prediction,i[:-overlap],overlapRegion))
            overlapTail = i[(maxNumWords-overlap):] #Extract tail n words for next iteration    
        
        return full_prediction

def splitSeq(seq, numWords, maxNumWords, overlap):
    X_trimmed = []
    X_trimmed.append(seq[0:maxNumWords]) #First case

    i = 1
    startPos = (maxNumWords-overlap)*i
    endPos = startPos + maxNumWords

    #Split until the end
    while endPos < numWords:
        X_trimmed.append(seq[startPos:endPos])
        #Update parameters
        i += 1
        startPos = (maxNumWords-overlap)*i
        endPos = startPos + maxNumWords

    #Last chunk
    X_trimmed.append(seq[startPos:numWords])

    return X_trimmed

def getTimestamps(transcript, captionCount, predictions, words, returnText = 0):
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
        if returnText:
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

    if returnText:
        return sponsorTimestamps, sponsorText
    else:
        return sponsorTimestamps
