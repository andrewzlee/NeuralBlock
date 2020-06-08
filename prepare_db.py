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
            "processed"     INTEGER NOT NULL
    )
    """
    cursor.execute(query)
    conn.commit()
    conn.close()
    return

def createRandom():
    conn = sqlite3.connect("data\labeled.db")
    cursor = conn.cursor()
    query2 = """
    CREATE TABLE "RandomData" (
            "videoID"       TEXT NOT NULL,
            "startTime"     REAL NOT NULL,
            "endTime"       REAL NOT NULL,
            "text" BLOB
    )
    """
    cursor.execute(query2)
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
            "sponsorLabel" BLOB
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


truncateTable("sponsordata")
truncateTable("randomdata")
truncateTable("sponsorstream")
#createSponsor()
#createRandom()
#createStream()