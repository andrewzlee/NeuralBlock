import sqlite3
from random import sample
from youtube_transcript_api import YouTubeTranscriptApi
import pafy

import traceback

def findBestSegments(cursor_src, vid, verbose = False):
    cursor_src.execute(f"select videoid, starttime, endtime, votes from sponsortimes where videoid = '{vid}' and votes > 1 order by votes desc")

    sponsors = []
    for i in cursor_src.fetchall():
        sponsors.append((i[1],i[2],i[3]))

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

def appendSponsor(conn_src, conn_dest, vid, verbose = False):
    try:
        cursor_dest = conn_dest.cursor()
        count = cursor_dest.execute(f"select count(*) from sponsordata where videoid = '{vid}'").fetchone()[0]
        if count > 0: #ignore if already in the db
            return
        
        best = findBestSegments(conn_src.cursor(), vid, verbose)
        transcript = YouTubeTranscriptApi.get_transcript(vid, languages=["en"])
        
        fuzziness = 0.25 #quarter of a second buffer
        #Map the time stamps to text
        for b in best:
            segLength = b[1] - b[0]
            if segLength/60 > 8: #only check sponsors >8 minutes to verify it's legitimate since it's computationally expensive
                totalLength = pafy.new(f"https://www.youtube.com/watch?v={vid}").length
                if segLength/totalLength >= 0.4: #ignore "sponsors" that are longer than 40% of the video
                    if verbose:
                        print(f"({vid}) has {round(segLength/60,2)} min sponsor out of a total {round(totalLength,2)} min video.")
                    cursor_dest.execute(f"insert into sponsordata values ('{vid}', {b[0]}, {b[1]}, {b[2]}, null , -1)")
                    continue
            string = ""
            for t in transcript:
                #fuzziness widens the labeled range
                if (b[0] - fuzziness) <= t["start"] <= (b[1] + fuzziness) :
                    string = string + t["text"].replace("\n"," ") + " "
            if verbose:
                print("SPONSOR::")
                print(string)
            string = string.replace("'", "''")
            cursor_dest.execute(f"insert into sponsordata values ('{vid}', {b[0]}, {b[1]}, {b[2]}, '{string}', 1)")
    except:
        #print(f"Video {vid} could not have its transcript extracted.") #No transcripts available or not English.
        for b in best:    
            cursor_dest.execute(f"insert into sponsordata values ('{vid}', {b[0]}, {b[1]}, {b[2]}, null , -1)")
    finally:
        conn_dest.commit()
        return

def appendRandom(conn_dest, verbose = False):
    cursor_dest = conn_dest.cursor()
    cursor_dest.execute(f"select endtime - starttime from sponsordata where processed = 1")
    segment_lengths = cursor_dest.fetchall()
    cursor_dest.execute(f"select distinct videoid from sponsordata where processed = 1")
    video_list = cursor_dest.fetchall()
    #fuzziness = 0.10 #quarter of a second buffer
    
    cnt = 1
    for vid in video_list:
        cursor_dest.execute(f"select starttime, endtime from sponsordata where videoid = '{vid[0]}'")
        best = cursor_dest.fetchall()
        
        transcript = YouTubeTranscriptApi.get_transcript(vid[0], languages=["en"])
        if len(transcript) < 5:
            print("Too little text. Skipping...")
            continue
        #Removes bug where last element is garbage
        if transcript[-1]["start"] < transcript[-2]["start"]:
            del transcript[-1] 
        
        selected_segments,start_used = [], []
        for i in range(0,len(best)+1):
            segment = sample(segment_lengths,1)[0][0] #length of video to extract
            flag, skip = True, False
            loopCounter, resampleCounter = 0, 0
            while flag:
                flag = False
                start_point = sample(transcript,1)[0]["start"]
                end_point = start_point + segment
                for b in best:
                    #If we selected a segment that is in a sponsorship OR if the segment is longer
                    #than the full video, we want to resample the start point OR if we've already used
                    #this section of video
                    if (b[1] > start_point > b[0] or b[1] > end_point > b[0] or 
                        end_point > (transcript[-1]["start"] + transcript[-1]["duration"]) or
                        start_point in start_used): 
                        flag = True
                loopCounter += 1
                if loopCounter % 100 == 0:
                    resampleCounter += 1
                    #If the segment length is causing an infinte loop resample
                    segment = sample(segment_lengths,1)[0][0] 
                    print(f"Resampling attempt {resampleCounter} of 20 on {vid[0]} {cnt} of {len(video_list)}")
                    if resampleCounter == 20:
                        print("Resampled 20 times. Moving on...")
                        skip = True
                        break
            if not skip:
                selected_segments.append((start_point, end_point,segment))
                start_used.append(start_point)
        
        for sel in selected_segments:
            string = ""
            for t in transcript:
                if sel[0] <= t["start"] <= sel[1]:
                    string = string + t["text"].replace("\n"," ") + " "
            
            if verbose and cnt % 500 == 0:
                print(f"('{vid[0]}', {sel[0]}, {sel[1]}, '{string}')\n")
                
            string = string.replace("'", "''")
            cursor_dest.execute(f"insert into randomdata values ('{vid[0]}', {sel[0]}, {sel[1]}, '{string}')")
            conn_dest.commit()
        if verbose and cnt % 100 == 0:
                print("Video ({}) {} of {}".format(vid[0], cnt, len(video_list)))
        cnt += 1
    return 

try:
    conn_src = sqlite3.connect(r"C:\Users\Andrew\Documents\NeuralBlock\data\database.db")
    conn_dest = sqlite3.connect(r"C:\Users\Andrew\Documents\NeuralBlock\data\labeled.db")
    
    cursor_src = conn_src.cursor()
    cursor_src.execute("select distinct videoid from sponsortimes where votes > 1")
    videoList = cursor_src.fetchall()
    
    i = 0
    for vid in videoList:
        i += 1
        if i % 500 == 0:
            print("Video ({}) {} of {}".format(vid[0], i,len(videoList)))
            appendSponsor(conn_src, conn_dest, vid[0], verbose = True)
        else:
            appendSponsor(conn_src, conn_dest, vid[0])
    
    #Part of the reason this is done separately is because 1) I wrote this
    #piece afterwards, and 2) it makes it possible to sample from the entire
    #distribution of segment lengths.
    appendRandom(conn_dest, verbose = True)
            
except:
    traceback.print_exc()
finally:
    print("Connection closed")
    conn_src.close()
    conn_dest.close()

