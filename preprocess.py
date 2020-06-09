import sqlite3
import random
from youtube_transcript_api import YouTubeTranscriptApi
import pafy
from tensorflow.keras.preprocessing.text import Tokenizer
import re
import traceback

def findBestSegments(cursor_src, vid, verbose = False):
    cursor_src.execute(f"select videoid, starttime, endtime, votes from sponsortimes where videoid = '{vid}' and votes > 1 order by votes desc")

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
    if verbose:
            print(best)
    return best

def extractText(b, transcript, widen = 0.1):
    wps = 3
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

def extractSponsor(conn_src, conn_dest, vid, best, transcript, verbose = False):
    try:
        cursor_dest = conn_dest.cursor()
        count = cursor_dest.execute(f"select count(*) from sponsordata where videoid = '{vid}'").fetchone()[0]
        if count > 0: #ignore if already in the db
            return
        
        #Check to see if the text even matches... overlapping text xEIt4OojA3Y

        wps = 2.25
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
                    cursor_dest.execute(f"insert into sponsordata values ('{vid}', {b[0]}, {b[1]}, {b[2]}, null , -1)")
                    continue
            
            string,totalNumWords = extractText(b,transcript)
            
            filledin = 0
            if totalNumWords < round(segLength*wps):
                transcript_list = YouTubeTranscriptApi.list_transcripts(vid)
                auto = transcript_list.find_generated_transcript(["en"])
                transcript_auto = auto.fetch()
                string, totalNumWords = extractText(b, transcript_auto)
                filledin = 1
                print(f"The totalNumWords count for ({vid}) is lower than expected. Pulled autogenerated text.")
            
            autogen = 1 #Hardcoded
            if verbose:
                print("SPONSOR::")
                print(string)
            string = string.replace("'", "''")
            #vid, text, sponsorship, autogen, filledin, processed/skipped
            cursor_dest.execute(f"insert into sponsordata values ('{vid}', {b[0]}, {b[1]}, '{string}', 1, {autogen}, {filledin}, 1)")
            
    except:
        print("Failed to extract sponsor.")
    finally:
        conn_dest.commit()
        return

def extractRandom(conn_dest, vid, best, transcript, verbose = False):
    #We're going to extract a random text excerpt from the transcript that
    #has the same length as our sponsored segments.
    cursor_dest = conn_dest.cursor()
        
    if len(transcript) < 5:
        print("Too little text. Skipping...")
        return
    
    #Removes bug where last element is garbage
    if transcript[-1]["start"] < transcript[-2]["start"]:
        del transcript[-1] 
    
    selected_segments,start_used = [], []
    for b in best:
        #Add some randomness with a minimum of 3 seconds
        segment = max(3, random.uniform(0.5,2.0) * (b[1]-b[0]))
        
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
                segment = max(3, random.uniform(0.5,1.5) * (b[1]-b[0]))
                print(f"Resampling attempt {resampleCounter} of 20 on {vid[0]}")
                if resampleCounter == 20:
                    print("Resampled 20 times. Moving on...")
                    skip = True
                    break
        if not skip:
            selected_segments.append((start_point, end_point,segment))
            start_used.append(start_point)
    
    #Simplified version of extractText()
    for sel in selected_segments:
        string = ""
        for t in transcript:
            if sel[0] <= t["start"] <= sel[1]:
                string = string + t["text"].replace("\n"," ") + " "
        
        if verbose:
            print(f"('{vid[0]}', {sel[0]}, {sel[1]}, '{string}')\n")
        
        string = string.replace("'", "''")
        autogen = 1
        cursor_dest.execute(f"insert into sponsordata values ('{vid}', {b[0]}, {b[1]}, '{string}', 0, {autogen}, null, 1)")
        conn_dest.commit()
    return 

def getWordCount(text):
    #Use tokenizer to be consistent with training method
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([text])
    text = tokenizer.texts_to_sequences([text])
    return len(text[0])

def appendData(full_text, seq, text, tStart, tEnd, best, verbose = False):
    full_text += re.sub(" +", " ", text) + " "
    numWords = getWordCount(text)
    
    inSponsor = False
    fuzziness = 0.100
    for b in best:
        if (b[0] - fuzziness) <= tEnd and tStart <= (b[1] + fuzziness):
            inSponsor = True
    
    if inSponsor:
        seq += [1] * len(numWords)
        if verbose:
            print(text)
    else: 
        seq += [0] * len(numWords)
        
    return full_text, seq


def labelVideo(conn_dest, vid, best, transcript, filledin, verbose = False):
    try:
        
        if filledin:
            transcript_list = YouTubeTranscriptApi.list_transcripts(vid)
            auto = transcript_list.find_generated_transcript(["en"])
            transcript_auto = auto.fetch()
        
        #Stitch together the transcript into a single string
        #Use the tokenized string to label each word as sponsor or not
        seq = []
        full_text = ""
        for t in transcript:
            tStart = t["start"]
            tEnd = tStart + t["duration"]
            
            if filledin:
                for b in best:
                    if b[0] <= tStart:
                        string, totalNumWords = extractText(b, transcript_auto) 
                        full_text, seq = appendData(full_text, seq, string, tStart, tEnd, best)
            
            raw_text = t["text"].replace("\n"," ")
            raw_text = re.sub(" +", " ", raw_text.replace(r"\u200b", " ")) #strip out this unicode
            full_text, seq = appendData(full_text, seq, raw_text, tStart, tEnd, best)
        
        for b in best:
            if b[0] > transcript[-1]["start"]:
                tStart = transcript[-1]["start"]
                tEnd = tStart + transcript[-1]["duration"]
                string, totalNumWords = extractText(b, transcript_auto) 
                full_text, seq = appendData(full_text, seq, string, tStart , tEnd, best)
        
        full_text = re.sub(" +", " ", full_text).replace("'", "''") #format text
        
        #insert text and labels into db
        cursor = conn_dest.cursor()
        cursor.execute(f"insert into SponsorStream values ('{vid}', '{full_text}' , '{seq}')")
        conn_dest.commit()
        
    except:
        #print(traceback.print_exc())
        print(f"{vid} failed to get subtitles.")
    return 

########
#Warning: Do not run this whole script at once. Each part was built independently
#and was run at different points in time. Specifically, the labelVideo() function
#pulls from sponsordata to create its own data.


if __name__ == "__main__":
    try:
        conn_src = sqlite3.connect(r"C:\Users\Andrew\Documents\NeuralBlock\data\database.db")
        conn_dest = sqlite3.connect(r"C:\Users\Andrew\Documents\NeuralBlock\data\labeled.db")
        
        cursor_src = conn_src.cursor()
        cursor_src.execute("select distinct videoid from sponsortimes where votes > 1")
        videoList = cursor_src.fetchall()
        
        #Extracts the text for a sponsor segment and labels it 1 (sponsor)
        i = 0
        for vid in videoList:
            i += 1
            try:
                transcript = YouTubeTranscriptApi.get_transcript(vid, languages=["en"])
                
                if i % 500 == 0:
                    print("Video ({}) {} of {}".format(vid[0], i,len(videoList)))
                    
                    best = findBestSegments(conn_src.cursor(), vid, verbose = True)
                    
                    extractSponsor(conn_src, conn_dest, vid[0], best, transcript, verbose = True)
                    extractRandom(conn_dest, best, transcript, verbose = True)
                    labelVideo(conn_dest, vid[0], verbose = True)
                    
                else:
                    best = findBestSegments(conn_src.cursor(), vid)
                    
                    extractSponsor(conn_src, conn_dest, vid[0], best, transcript)
                    extractRandom(conn_dest, best, transcript)
                    labelVideo(conn_dest, vid[0])
            except:
                #print(f"Video {vid} could not have its transcript extracted.") #No transcripts available or not English.
                for b in best:
                    cursor_dest = conn_dest.cursor()
                    cursor_dest.execute(f"insert into sponsordata values ('{vid}', {b[0]}, {b[1]}, null, 1, null, null, 0)")
            
            if i == 3:
                break;
                
        ##################################################################
        
        #Labels the sponsored segments for the whole video. It uses some of the 
        #data computed above to save time mainly.
        cur = conn_dest.cursor()
        cur.execute("select distinct videoid from sponsordata where processed = 1")
        videoList = cur.fetchall()
        i = 0
        for vid in videoList:
            i+=1
            if i % 500 == 0:
                print("Video ({}) {} of {}".format(vid[0], i,len(videoList)))
                labelVideo(conn_dest, vid[0], verbose = True)
            else:
                labelVideo(conn_dest, vid[0])
                
    except:
        traceback.print_exc()
    finally:
        print("Connection closed")
        conn_src.close()
        conn_dest.close()
        