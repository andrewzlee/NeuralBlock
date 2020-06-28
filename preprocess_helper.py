# -*- coding: utf-8 -*-
from youtube_transcript_api import YouTubeTranscriptApi
import random
import pafy
from tensorflow.keras.preprocessing.text import Tokenizer
import re

def findBestSegments(cursor_src, vid, verbose):
    cursor_src.execute(f"select videoid, starttime, endtime, votes from sponsortimes where videoid = '{vid}' and category = 'sponsor' and votes > 0 order by votes desc")

    sponsors = []
    for i in cursor_src.fetchall():
        sponsors.append((i[1],i[2],i[3]))
    
    #Ported algorithm originally written by Ajay for his SponsorBlock project
    #Find sponsors that are overlapping
    similar = []
    for i in sponsors:
        for j in sponsors:
            if (j[0] >= i[0] and j[0] <= i[1]):
                similar.append([i,j])
    
    #Within each group, choose the segment with the most votes.
    dealtWithSimilarSponsors = []
    best = []
    for i in similar:
        if i in dealtWithSimilarSponsors:
            continue
        group = i
        for j in similar:
            if j[0] in group or j[1] in group:
                group.append(j[0])
                group.append(j[1])
                dealtWithSimilarSponsors.append(j)
        best.append(max(set(group), key = lambda item:item[2]))
    
    best.sort()
    if verbose:
            print(best)
    return best

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
            
    return string, totalNumWords

def extractSponsor(conn_dest, vid, best, transcript, autogen, verbose):
    status = 1
    isManual = not autogen
    
    cursor_dest = conn_dest.cursor()
    count = cursor_dest.execute(f"select count(*) from sponsordata where videoid = '{vid}'").fetchone()[0]
    if count > 0: #ignore if already in the db
        print(f"Already been lableled ({vid}).")
        return 0
    
    #Check to see if the text even matches... overlapping text xEIt4OojA3Y

    #Map the time stamps to text
    for b in best:
        segLength = b[1] - b[0]
        lenThreshold = 7
        if segLength/60 > lenThreshold: #only check sponsors >7 minutes to verify it's legitimate since it's computationally expensive
            print(f"Sponsor in ({vid}) is longer than {lenThreshold} minutes. Checking to make sure it's okay.")
            totalLength = pafy.new(f"https://www.youtube.com/watch?v={vid}").length
            if segLength/totalLength >= 0.45: #ignore "sponsors" that are longer than 45% of the video
                if verbose:
                    print(f"({vid}) has {round(segLength/60,2)} min sponsor out of a total {round(totalLength,2)} min video.")
                cursor_dest.execute(f"insert into sponsordata values ('{vid[0]}', {b[0]}, {b[1]}, null, 1, null, null, 0, current_date)")
                status = 0
                continue
        
        string,totalNumWords = extractText(b,transcript)
        
        filledIn = 0
        #When a segment has less than the usual words, use the autogen
        #if avaiable. If already using autogen, just check to see if it's
        #silence or un-transcribe-able and fail if true. Otherwise, let it 
        #slide and warn in the log.
        expWords = segLength*2.3
        if totalNumWords < expWords:
            if isManual:
                try:
                    print(f"The totalNumWords count for ({vid}) is lower than expected. Pulling autogenerated text.")
                    transcript_list = YouTubeTranscriptApi.list_transcripts(vid)
                    auto = transcript_list.find_generated_transcript(["en"])
                    transcript_auto = auto.fetch()
                    string, totalNumWords = extractText(b, transcript_auto)
                    filledIn = 1
                except:
                    print(f"Failed to pull autogenerated text for ({vid}).")
                    cursor_dest.execute(f"insert into sponsordata values ('{vid[0]}', {b[0]}, {b[1]}, null, 1, null, null, 0, current_date)")
                    status = 0
                    continue
            elif totalNumWords < segLength*0.75: #if autogen is mostly silent
                print(f"Autogen for ({vid}) has too few words.")
                cursor_dest.execute(f"insert into sponsordata values ('{vid[0]}', {b[0]}, {b[1]}, null, 1, null, null, 0, current_date)")
                status = 0
                continue
            else:
                print(f"Autogen for ({vid}) has a low count then expected ({totalNumWords}/{round(expWords)}) but continuing anyways.")
                print("[TEXT] " + string)
                
        if verbose:
            print("SPONSOR::")
            print(string)
        
        string = string.replace("'", "''")
        #vid, text, sponsorship, autogen, filledin, processed/skipped
        cursor_dest.execute(f"insert into sponsordata values ('{vid}', {b[0]}, {b[1]}, '{string}', 1, {autogen}, {filledIn}, 1, current_date)")
        conn_dest.commit()
    return filledIn, status

def extractRandom(conn_dest, vid, best, transcript, autogen, verbose):
    #We're going to extract a random text excerpt from the transcript that
    #has the same length as our sponsored segments.    
    if len(transcript) < 5:
        print("Too little text. Skipping...")
        return
    
    #Removes bug where last element is garbage
    if transcript[-1]["start"] < transcript[-2]["start"]:
        del transcript[-1] 
    
    selected_segments,start_used = [], []
    for b in best:
        #Add some randomness with a minimum of 3 seconds
        segment = max(3, round(random.uniform(0.5,2.0) * (b[1]-b[0]),2))
        
        flag, skip = True, False
        loopCounter, resampleCounter = 0, 0
        while flag:
            flag = False
            start_point = random.sample(transcript,1)[0]["start"]
            end_point = start_point + segment
            for b in best:
                #If we selected a segment that overlaps with a sponsorship OR if the segment extends
                #past the end of the video, we want to resample the start point OR if we've already used
                #this section of video
                if ((b[0] <= end_point and start_point <= b[1]) or 
                    end_point > (transcript[-1]["start"] + transcript[-1]["duration"]) or
                    start_point in start_used): 
                    flag = True 
            loopCounter += 1
            if loopCounter % 100 == 0:
                resampleCounter += 1
                #If the segment length is causing an infinte loop resample
                segment = max(3, round(random.uniform(0.5,2.0) * (b[1]-b[0]),2))
                print(f"Resampling attempt {resampleCounter} of 20 on {vid}")
                if resampleCounter == 20:
                    print("Resampled 20 times. Moving on...")
                    skip = True
                    break
        if not skip:
            selected_segments.append((start_point, end_point, segment))
            start_used.append(start_point)
    
    #Simplified version of extractText()
    for sel in selected_segments:
        string = ""
        for t in transcript:
            if sel[0] <= t["start"] <= sel[1]:
                string = string + t["text"].replace("\n"," ") + " "
        
        if verbose:
                print(f"RANDOM:: ({sel[0]}, {sel[1]})")
                print(string)
        
        string = string.replace("'", "''")
        cursor_dest = conn_dest.cursor()
        cursor_dest.execute(f"insert into sponsordata values ('{vid}', {sel[0]}, {sel[1]}, '{string}', 0, {autogen}, null, 1, current_date)")
        conn_dest.commit()
    return 

def getWordCount(text):
    #Use tokenizer to be consistent with training method
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([text])
    text_seq = tokenizer.texts_to_sequences([text])
    return len(text_seq[0])

def appendData(full_text, seq, text, tStart, tEnd, best, autogen, verbose):
    full_text += re.sub(" +", " ", text) + " "
    numWords = getWordCount(text)
    inSponsor = False
    fuzziness = 0.200 #200ms
    for b in best:
        if (b[0] - fuzziness) <= tEnd and tStart <= (b[1] + fuzziness):
            inSponsor = True
            # sponsorTimes = (b[0] - fuzziness, (b[1] + fuzziness))
    
    if inSponsor or autogen:
        # Attempt to find a more accurate start and stop point for labeling.
        # Wasn't successful so we'll have to stick with extra text 
        # at the start and end ¯\_(ツ)_/¯
        
        # wps = 2.3
        # excessHead = round((sponsorTimes[0]-tStart)*wps) #how many words can we cut out?
        # excessTail = round((tEnd - sponsorTimes[1])*wps) #how many words to keep?
        # startLoc = min(max(excessHead,0), numWords) #2 word buffer
        # #theoretical words - actual words
        # silentWords = round((tEnd-tStart)*wps) - numWords
        # endLoc = max(excessTail-silentWords, 0)
        
        # seq += [0] * startLoc
        # seq += [1] * (numWords-startLoc)#-endLoc)
        # seq += [0] * endLoc
        
        seq += [1] * numWords 
        
        if verbose:
            # words = text.split()
            # string = words[startLoc:]#(numWords-endLoc)]
            # if len(string) > 0: 
            #     print(" ".join(string))
            print(text)
            
    else:
        seq += [0] * numWords
        
    return full_text, seq


def labelVideo(conn_dest, vid, best, transcript, filledIn, autogen, verbose):
    print(f"filled in status is: {filledIn}")
    if filledIn: #Must have been manual transcript, so we need the autogen
        transcript_list = YouTubeTranscriptApi.list_transcripts(vid)
        transcript_auto = transcript_list.find_generated_transcript(["en"]).fetch()
    
    #Stitch together the transcript into a single string
    #Use the tokenized string to label each word as sponsor or not
    seq = []
    full_text = ""
    segList = best.copy()
    for t in transcript:
        tStart = t["start"]
        tEnd = tStart + t["duration"]
        
        if filledIn:
            for b in segList:
                if b[0] <= tStart:
                    string, totalNumWords = extractText(b, transcript_auto) 
                    full_text, seq = appendData(full_text, seq, string, tStart, tEnd, best, 1, verbose)
                    segList.remove((b[0],b[1],b[2]))
        
        raw_text = t["text"].replace("\n"," ")
        raw_text = re.sub(" +", " ", raw_text.replace(r"\u200b", " ")) #strip out this unicode
        full_text, seq = appendData(full_text, seq, raw_text, tStart, tEnd, best, 0, verbose)
    
    if filledIn:
        for b in segList:
            if b[0] > transcript[-1]["start"]:
                tStart = transcript[-1]["start"]
                tEnd = tStart + transcript[-1]["duration"]
                string, totalNumWords = extractText(b, transcript_auto) 
                full_text, seq = appendData(full_text, seq, string, tStart , tEnd, best, 1, verbose)
        
    full_text = re.sub(" +", " ", full_text).replace("'", "''") #format text
    
    #insert text and labels into db
    cursor = conn_dest.cursor()
    cursor.execute(f"insert into SponsorStream values ('{vid}', '{full_text}' , '{seq}', {autogen}, {filledIn}, 1, current_date)")
    conn_dest.commit()
        
    return 

def labelData(conn_dest, vid, best, transcript, useAutogen, verbose):
    filledIn, status = extractSponsor(conn_dest, vid, best, transcript, useAutogen, verbose)
    if status:
        extractRandom(conn_dest, vid, best, transcript, useAutogen, verbose)
        labelVideo(conn_dest, vid, best, transcript, filledIn, useAutogen, False)
    return status

def insertBlanks(conn_dest, cursor_dest, best, vid):
    for b in best:
        #print(f"Video {vid} could not have its transcript extracted.") #No transcripts available or not English.
        cursor_dest.execute(f"insert into sponsordata values ('{vid}', {b[0]}, {b[1]}, null, 1, null, null, 0, current_date)")
        conn_dest.commit()
    return