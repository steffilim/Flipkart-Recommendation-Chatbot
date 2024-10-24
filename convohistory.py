import os
from dotenv import load_dotenv

from pymongo import MongoClient

from typing import List, Tuple
import datetime
from functions import initialising_mongoDB

load_dotenv()


""" CONVERSATION HISTORY FOR REGISTERED USERS """
db = initialising_mongoDB()
chat_session = db.chatSession


def start_new_session(user_id, session_id):
    document = {
        "user_id": user_id,
        "session_id": session_id,
        "created_at": datetime.datetime.now(),
        "message_list": []
    }
    chat_session.insert_one(document)
    
def add_chat_history_user(session_id, user_input,user_intention, bot_response):
    dict = {"user_input": user_input, "user_intention": user_intention, "bot_response": bot_response}
    chat_session.update_one({"session_id": session_id}, {"$push": {"message_list": dict}})
    print("Chat history updated successfully")

def get_past_conversations_users(user_id,session_id):

    response = chat_session.find_one({"user_id": user_id, "session_id": session_id})
    if response is None:
        return ""
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