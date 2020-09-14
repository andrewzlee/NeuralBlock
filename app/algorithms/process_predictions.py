from tensorflow.keras.preprocessing.sequence import pad_sequences
from youtube_transcript_api import YouTubeTranscriptApi
import re
import numpy as np
import pandas as pd
#import ds_stt as stt

#Copied from preprocess.py
def extractText(b, transcript, widen = 0.150):
    wps = 2.3
    totalNumWords = 0
    string = ""

    for t in transcript:
        tStart = t["start"]
        tEnd = tStart + t["duration"]

        text = t["text"].split()
        numWords = len(text)

        #Store a wider range for the labeled text
        if (b[0] - widen) <= tEnd and tStart <= (b[1] + widen):
            totalNumWords += numWords
            excessHead = round((b[0]-tStart)*wps) #how many seconds can we cut out?
            excessTail = round((b[1]-tStart)*wps) #how many words to keep?

            clean_txt = t["text"].replace("\n"," ").split()
            startLoc = max(excessHead,0)
            endLoc = min(excessTail, numWords)
            string = string + " ".join(clean_txt[startLoc:endLoc]) + " "

    return string, len(string.split())

def getPredictionsSpot(model,tokenizer,vid,segments):
    processed = []
    text = []
    try:
        #Pull transcript list
        transcript_list = YouTubeTranscriptApi.list_transcripts(vid)

        for seg in segments:
            expWords = (seg[1]-seg[0])*2.3
            try:
                #Pull manual transcript
                transcript_manual = transcript_list.find_manually_created_transcript(["en","en-GB"]).fetch()
                string,totalNumWords = extractText(seg, transcript_manual, widen = 0.05)

                if totalNumWords < expWords*0.65:
                    raise("Too few words. Try autogen.")
                else:
                    text.append(string)
                    processed.append(0)
            except:
                try:
                    #Pull autogen if manual is too low or doesn't exist
                    transcript_auto = transcript_list.find_generated_transcript(["en"]).fetch()
                    string,totalNumWords = extractText(seg, transcript_auto, widen = 0.05)
                    if totalNumWords < expWords*0.65:
                        processed.append(1)
                    else:
                        text.append(string)
                        processed.append(0)
                except:
                    # Video has no autogen
                    return [1.0] * len(segments) , 422
    except:
        # Video has no transcripts
        return [1.0] * len(segments), 422

        #text.append(string)
    if len(text) > 0:
        data = pd.DataFrame({"text":text})
        x_new = tokenizer.texts_to_sequences(data["text"].values)
        x_new = pad_sequences(x_new, padding = "post", maxlen = 3000, truncating = "post")

        nb_predictions = model.predict(x_new, batch_size = len(text))[:,1].tolist() #P(Sponsor|text)

    nbIdx = 0
    predictions = []
    for p in processed:
        if p:
            predictions.append(1.0)
        else:
            predictions.append(nb_predictions[nbIdx])
            nbIdx += 1
    return predictions, 200


def processVideoStream(vid, useDS = False):

    #Use DeepSpeech or YoutubeTranscriptApi to get transcript
    if useDS:
        #stt.yt_download(vid)
        #transcript, fullText = stt.transcribe(vid + ".wav")
        #captionCount = [1] * len(transcript)
        print("DS is not ready yet.")

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

def getPredictionsStream(model,tokenizer,text):
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

def getTimestampsStream(transcript, captionCount, predictions, words, returnText = 0):
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
