import os
import psycopg2
from dotenv import load_dotenv
 
load_dotenv()
 
DB_HOST = os.getenv("DB_HOST")
DB_NAME = "chatbot_history_db"  
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

def connect_db():
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

def add_chat_history(user_id, user_input, bot_response):
    conn = connect_db()
    if conn is not None:
        try:
            cursor = conn.cursor()
            insert_query = """
            INSERT INTO chat_history (user_id, user_input, bot_response) 
            VALUES (%s, %s, %s);
            """
            cursor.execute(insert_query, (user_id, user_input, bot_response))
            conn.commit()
            cursor.close()
        except Exception as e:
            print(f"Error inserting chat history: {e}")
        finally:
            conn.close()


def get_past_conversations(user_id):
    conn = connect_db()
    if conn is not None:
        try:
            cursor = conn.cursor()
            query = "SELECT user_input, bot_response FROM chat_history WHERE user_id = %s"
            cursor.execute(query, (user_id,))
            history = cursor.fetchall()
            cursor.close()
            return history
        except Exception as e:
            print(f"Error retrieving chat history: {e}")
        finally:
            conn.close()
    return []
