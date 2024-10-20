import os
from dotenv import load_dotenv

from pymongo import MongoClient
from gemini import app
from typing import List, Tuple
import datetime

load_dotenv()

"""
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_API_KEY = os.getenv("SUPABASE_API_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_API_KEY)

def add_chat_history_user(user_id, session_id, user_input, bot_response, user_intention):
    data = {
        "user_id": user_id,
        "session_id": session_id,  # save the session ID
        "user_input": user_input,
        "bot_response": bot_response,
        "intention": user_intention
    }
    response = supabase.table("chat_history").insert(data).execute()
    return response

def get_past_conversations_users(user_id, session_id):
    response = (supabase.table("chat_history")
                .select("intention")
                .eq("user_id", user_id)
                .eq("session_id", session_id)
                .order("created_at", desc=True) # sorting the conversations by the most recent interaction
                .limit(1) # getting the most recent interaction
                .execute())
    past_convo = response.data
    string = " ".join(d['intention'] for d in past_convo)

    return string
    
"""

""" CONVERSATION HISTORY FOR REGISTERED USERS """
ATLAS_URI = os.getenv("ATLAS_URI")
DB_NAME = os.getenv("DB_NAME")

client = MongoClient(ATLAS_URI)
DB_NAME = client[DB_NAME]
collection = DB_NAME["chatSession"]

def get_database():
    """Create a database connection and return the database client."""
    mongo_uri = os.getenv(ATLAS_URI)  # Ensure you have MONGO_URI in your .env file
    client = MongoClient(mongo_uri)
    return client[DB_NAME]

def start_new_session(user_id, session_id):
    document = {
        "user_id": user_id,
        "session_id": session_id,
        "created_at": datetime.datetime.now(),
        "message_list": []
    }
    collection.insert_one(document)
    
def add_chat_history_user(session_id, user_input, bot_response):
    dict = {"user_input": user_input, "bot_response": bot_response}
    collection.update_one({"session_id": session_id}, {"$push": {"message_list": dict}})
    print("Chat history updated successfully")

def get_past_conversations_users(session_id):
    response = collection.find_one({"session_id": session_id})
    past_convo = response["message_list"]
    string = " ".join(d['user_input'] for d in past_convo)

    return string



"""CONVERSATION HISTORY FOR GUEST USERS """
# function to update the chat history (which is stored in a list)
def add_chat_history_guest(user_input, bot_response, convo_history_list_guest):
    # Append the current user query and bot response as a tuple to the conversation history
    convo_history_list_guest.append((user_input, bot_response))


# function to get the past intention of the user
def get_past_conversation_guest(memory) -> List[str]:
    # Extract the last element from each tuple in the memory list
    last_elements = [msg[0] for msg in memory]  # msg[0] accesses the first element of each tuple
    
    return last_elements