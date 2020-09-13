# -*- coding: utf-8 -*-

#Misc queries to check data

import sqlite3
import random


conn_src = sqlite3.connect(r"./data/database.db")
conn_dest = sqlite3.connect(r"./data/labeled.db")


cursor_src = conn_src.cursor()
cursor_dest = conn_dest.cursor()


query = "select processed, autogen, count(distinct videoid) from sponsordata group by processed, autogen"
query = "select videoid from sponsordata where sponsor = 1 group by videoid, starttime, endtime having count(*) > 1"
query = "select * from sponsordata where sponsor = 1 and length(text) = 0"
#query = "select count(distinct videoid) from sponsortimes where votes > 1"
#query = "delete from sponsordata where videoid = 'xy-RUYg-xuQ'"

cursor_dest.execute(query)
#conn_dest.commit()
res = cursor_dest.fetchall()

for r in res:
    print(r)

conn_src.close()
conn_dest.close()
