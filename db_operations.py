import pymysql
import mysql.connector

# Database connection details
DB_HOST = "34.47.118.205"
DB_USER = "bbb"
DB_PASSWORD = "1234"
DB_NAME = "airplain_RDB"

def insert_data_bulk(values):
    print("Insert bulk data")
    insert_sql = """
    INSERT INTO flights
    (flight_number, airline, flight_date, STD, Departure, ARRIVAL, IO) 
    VALUES (%(flight_number)s, %(airline)s, %(flight_date)s, %(STD)s, %(Departure)s, %(ARRIVAL)s, %(IO)s)
    """
    conn = pymysql.connect(user=DB_USER, password=DB_PASSWORD, host=DB_HOST, database=DB_NAME)
    try:
        with conn.cursor() as cur:
            cur.executemany(insert_sql, values)
        conn.commit()
        print(f"Successfully inserted {len(values)} records")
    except Exception as e:
        print(f"Error inserting data: {e}")
        conn.rollback()
    finally:
        conn.close()

def select_data(query):
    mydb = mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME
    )
    try:
        with mydb.cursor() as mycursor:
            mycursor.execute(query)
            myresult = mycursor.fetchall()
            return myresult
    except Exception as e:
        print(f"Error executing query: {e}")
        return []
    finally:
        mydb.close()