import sqlite3
from sqlite3 import*

def sql_connection(): 
    try:
        con = sqlite3.connect('mydatabase.db')
        return con
 
    except Error:
        print(Error)

def sql_table(con):
        cursor = con.cursor()
        cursor.execute("""CREATE TABLE IF NOT EXISTS users(id integer PRIMARY KEY, name text)""")
        cursor.execute("""CREATE TABLE IF NOT EXISTS sensors(temp real, oxygen integer, humidity real, time_created text)""")
        con.commit()
    
 
 
