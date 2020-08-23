import sqlite3
import random
from youtube_transcript_api import YouTubeTranscriptApi
import preprocess_helper as preprocess
import traceback


def main():
    # Modes:
    #     1 Read from database.db and pull all videos 
    #     2 Read from labeled.db and pull only unprocessed
    #     3 Subtract labeled.db from database.db to pull newly labeled videos
    mode = 1
    
    try:
        conn_src = sqlite3.connect(r"./data/database.db")
        conn_dest = sqlite3.connect(r"./data/labeled.db")
        cursor_dest = conn_dest.cursor()
        
        if mode == 1:
            cursor_src = conn_src.cursor()
            cursor_src.execute("select videoid from sponsortimes where ((votes > 2) or (votes >= 0 and views >= 10)) and category = 'sponsor' and shadowHidden != 1")
            videoList = cursor_src.fetchall()
        elif mode ==  2:
            cursor_dest.execute("select distinct videoid from sponsordata where processed = 0")
            videoList = cursor_dest.fetchall()
        else: #Mode 3
            cursor_src = conn_src.cursor()
            cursor_src.execute("select videoid from sponsortimes where ((votes > 2) or (votes >= 0 and views >= 10)) and category = 'sponsor' and shadowHidden != 1")
            db_list = cursor_src.fetchall()
            
            cursor_dest.execute("select distinct videoid from sponsordata")
            lb_list = cursor_dest.fetchall()
            #Subtract processed videos out to find new videos
            videoList = list(set(db_list) - set(lb_list))
            
        
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
            assert i == (manCount + autoCount + skipCount), "Count mismatch for VideoID {}: {}".format(i,vid)
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