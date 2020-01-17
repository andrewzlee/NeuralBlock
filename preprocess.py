
import pandas as pd
from youtube_transcript_api import YouTubeTranscriptApi
import sqlite3

"""
CREATE TABLE "sponsorTimes" (
        "videoID"       TEXT NOT NULL,
        "startTime"     REAL NOT NULL,
        "endTime"       REAL NOT NULL,
        "votes" INTEGER NOT NULL,
        "UUID"  TEXT NOT NULL UNIQUE,
        "userID"        TEXT NOT NULL,
        "timeSubmitted" INTEGER NOT NULL,
        "views" INTEGER NOT NULL,
        "shadowHidden"  INTEGER NOT NULL
)
"""

def appendSponsor(cursor, df, vid, verbose = False):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(vid, languages=["en"])
        cursor.execute(f"select videoid, starttime,endtime,votes from sponsortimes where videoid = '{vid}' order by votes desc")
    
        sponsors = []
        for i in cursor.fetchall():
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
        
        fuzziness = 0.25 #quarter of a second buffer
        #Map the time stamps to text
        for b in best:
            string = ""
            for t in transcript:
                #fuzziness shrinks the labeled range
                if t["start"] > b[0] + fuzziness and t["start"] < b[1] - fuzziness:
                    string = string + t["text"].replace("\n"," ") + " "
            if verbose:
                print("SPONSOR::")
                print(string)
            df = df.append({"VID":vid, "START" : b[0], "END" : b[1], "VOTES":b[2], "TEXT":string}, ignore_index=True)
    except:
        print(f"Video {vid} could not have its transcript extracted.") #No transcripts allowed or not English.
    finally:
        return df

try:
    conn = sqlite3.connect(r"C:\Users\Andrew\Documents\sponsor_blocker\database.db")
    cursor = conn.cursor()
    
    cursor.execute("select distinct videoid from sponsortimes")
    videoList = cursor.fetchall()
    
    df = pd.DataFrame(columns = ["VID","START", "END", "VOTES", "TEXT"])
    
    i = 0
    for vid in videoList:
        i += 1
        if i % 500 == 0:
            print("Video {} of {}".format(i,len(videoList)))
            print(vid[0])
            df = appendSponsor(cursor, df, vid[0], verbose = True)
        else:
            df = appendSponsor(cursor, df, vid[0])
except Exception as e:
    print(e)
finally:
    print("Connection closed")
    conn.close()    

