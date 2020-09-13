import sqlite3

#Queries to create prepare DBs. Used before preprocess.py

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

def createSponsor():
    conn = sqlite3.connect("data/labeled.db")
    cursor = conn.cursor()
    query = """
    CREATE TABLE "SponsorData" (
            "videoID"       TEXT NOT NULL,
            "startTime"     REAL NOT NULL,
            "endTime"       REAL NOT NULL,
            "text"          BLOB,
            "sponsor"       INTEGER NOT NULL,
            "autogen"       INTEGER,
            "filledin"      INTEGER,
            "processed"     INTEGER NOT NULL,
            "dateprocessed" DATE NOT NULL
    )
    """
    cursor.execute(query)
    conn.commit()
    conn.close()
    return

def createStream():
    conn = sqlite3.connect("data/labeled.db")
    cursor = conn.cursor()
    query = """
    CREATE TABLE "SponsorStream" (
            "videoID"       TEXT NOT NULL,
            "text" BLOB,
            "sponsorLabel" BLOB,
            "autogen"       INTEGER,
            "filledin"      INTEGER,
            "processed"     INTEGER NOT NULL,
            "dateprocessed" DATE NOT NULL
    )
    """
    cursor.execute(query)
    conn.commit()
    conn.close()
    return

def truncateTable(table):
    conn = sqlite3.connect("data/labeled.db")
    cursor = conn.cursor()
    cursor.execute(f"delete from {table}")
    conn.commit()
    cursor.execute("vacuum")
    conn.close()
    return


#truncateTable("sponsordata")
#truncateTable("sponsorstream")
createSponsor()
createStream()
