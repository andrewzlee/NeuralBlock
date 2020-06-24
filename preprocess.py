import sqlite3
import random
from youtube_transcript_api import YouTubeTranscriptApi
import preprocess_helper as preprocess
import traceback


def main():
    # Modes:
    #     1 Read from database.db and pull all videos 
    #     2 Read from labeled.db and pull only unprocessed
    mode = 2
    
    try:
        conn_src = sqlite3.connect(r"./data/database.db")
        conn_dest = sqlite3.connect(r"./data/labeled.db")
        cursor_dest = conn_dest.cursor()        
        
        if mode == 1:
            query = "select distinct videoid from sponsortimes where votes > 1" 
            cursor_src = conn_src.cursor()
            cursor_src.execute(query)
            videoList = cursor_src.fetchall()
        else:
            query = "select distinct videoid from sponsordata where processed = 0"
            cursor_dest.execute(query)
            videoList = cursor_dest.fetchall()
        
        # LTT, Geo, HAI
        #videoList = [["xEIt4OojA3Y"]]#[["DdzwSM3HAuA"], ["xEIt4OojA3Y"], ["b00j2aCT6Ug"]]
        #videoList = [["yxEu8ucGUuE"]]
        #videoList = random.sample(videoList,100)
        
        #Build the datasets for normal inference and streaming inference.
        i = 1
        manCount = 0
        autoCount = 0
        skipCount = 0
        for vid in videoList:
            
            if mode == 2:
                cursor_dest.execute(f"delete from sponsordata where videoid = '{vid[0]}'")
                cursor_dest.execute(f"delete from sponsorstream where videoid = '{vid[0]}'")
                conn_dest.commit()
            
            #Print to console every 500 videos. 
            if i % 100 == 0:
                print("Video ({}) {} of {}".format(vid[0], i,len(videoList)))
                verbose = True
            else:
                verbose = False
            
            #Check for manual, then autogen and record which one is used.
            try:
                best = preprocess.findBestSegments(conn_src.cursor(), vid[0], verbose)
                transcript_list = YouTubeTranscriptApi.list_transcripts(vid[0])
                try:
                    useAutogen = 0
                    transcript_manual = transcript_list.find_manually_created_transcript(["en","en-GB"]).fetch()
                    status = preprocess.labelData(conn_dest, vid[0], best, transcript_manual, 
                              useAutogen, verbose)
                    if status:
                        manCount += 1
                    else:
                        skipCount += 1
                except:
                    try:
                        useAutogen = 1
                        transcript_auto = transcript_list.find_generated_transcript(["en"]).fetch()
                        status = preprocess.labelData(conn_dest, vid[0], best, transcript_auto, 
                                  useAutogen, verbose)
                        if status:
                            autoCount += 1
                        else:
                            skipCount += 1
                    except:
                        skipCount += 1
                        preprocess.insertBlanks(conn_dest, cursor_dest, best, vid[0])
                            
            except:
                skipCount += 1
                preprocess.insertBlanks(conn_dest, cursor_dest, best, vid[0])
            
            #Check to make sure labelData isn't being called more than
            #once per video.
            if i != (manCount + autoCount + skipCount):
                print("VideoID {}: {}".format(i,vid))
                raise Exception("Count mismatch.")
            i += 1
    except:
        traceback.print_exc()
    finally:
        print("Connection closed")
        cursor_dest.execute("vacuum")
        conn_src.close()
        conn_dest.close()
        print("Manual {}".format(manCount))
        print("Auto   {}".format(autoCount))
        print("Skip   {}".format(skipCount))
        
        
    return


if __name__ == "__main__":
    main()