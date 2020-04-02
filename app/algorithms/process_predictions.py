from tensorflow.keras.preprocessing.sequence import pad_sequences

def getPredictions(model,tokenizer,text):
    full_seq = tokenizer.texts_to_sequences([text])
    seqlen = len(full_seq[0])
    print("Sequence length: {}".format(seqlen))
    #Videos with more than 3000 words need to be split up.
    maxseqlen = 3000
    if seqlen <= maxseqlen:
        full_seq = pad_sequences(full_seq, maxlen = maxseqlen, padding = "post")
        return model.predict(full_seq, batch_size = 1).round(3) , 1
    else:
        print("Video needs to be 3000 words or fewer.")
    return [], 0

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
