# -*- coding: utf-8 -*-

import sqlite3



#conn_src = sqlite3.connect(r"./data/database.db")
conn_dest = sqlite3.connect(r"./data/labeled.db")


#cursor_src = conn_src.cursor()
cursor_dest = conn_dest.cursor()


query = "select * from sponsordata"

cursor_dest.execute(query)
res = cursor_dest.fetchall()

for r in res:
    print(r)
    

conn_dest.close()