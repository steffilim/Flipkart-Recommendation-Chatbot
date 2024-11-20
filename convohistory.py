import os
from dotenv import load_dotenv

import pymongo
from pymongo import MongoClient
from pymongo import DESCENDING

from typing import List, Tuple
import datetime
from functions.databaseFunctions import initialising_mongoDB

load_dotenv()


""" CONVERSATION HISTORY FOR REGISTERED USERS """
import re
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
    
def add_chat_history_user(session_id, user_input, user_intention_dictionary, items_recommended):
    # Build the dictionary to be pushed into the message list
    follow_up = user_intention_dictionary.get("Follow-Up Question")
    print("line 31: ", follow_up)
    message_entry = {
        "user_input": user_input,
        "user_intention": user_intention_dictionary,
        "items_recommended": items_recommended,
        "follow up": follow_up
    }

    # Attempt to update the document in MongoDB
    chat_session.update_one(
        {"session_id": session_id},  # Ensure this is the correct field and type in your MongoDB documents
        {"$push": {"message_list": message_entry}}
    )

    # Check the result of the update
    

def get_past_conversations_users(user_id,session_id):

    initialisation  = chat_session.find_one({"user_id": user_id, "session_id": session_id})
    if initialisation is None:
        return "", "", ""
    
    # getting the past user inputs
    past_chats = initialisation["message_list"]
    if len(past_chats) <= 0:
        return "", "", ""
    
    else:

        #past_user_inputs = " ".join(d['user_input'] for d in past_chats)
        previous_intention = past_chats[-1]["user_intention"]

        previous_follow_up_question = past_chats[-1]["follow up"]
        #previous_follow_up_match = re.search(r'- Follow-Up Question: (\w+)', previous_follow_up)
        #previous_follow_up_question = previous_follow_up_match.group(1)
        #print("last_bot_response: ", last_bot_response)
        previous_items_recommended = past_chats[-1]["items_recommended"]
   

        return previous_intention, previous_follow_up_question, previous_items_recommended



"""CONVERSATION HISTORY FOR GUEST USERS """
# function to update the chat history (which is stored in a list)
def add_chat_history_guest(user_input, user_intention_dictionary, recommendations, convo_history_list_guest):
    # Append the current user query and bot response as a tuple to the conversation history
    #print("Convo History List Guest before appending: ", convo_history_list_guest)
    print("line 78: ", recommendations)
    if user_intention_dictionary.get("Related to Follow-Up Questions") == "Old":
        #print(recommendations)
        follow_up_question = user_intention_dictionary.get("Follow-Up Question")
        convo_history_list_guest.append((user_input, user_intention_dictionary, recommendations, follow_up_question))
        convo_history_list_guest = convo_history_list_guest[1:] # removing the last element from the list
       
        
       

    else:
        convo_history_list_guest.append((user_input, user_intention_dictionary, recommendations))

    print("Convo History List Guest after appending follow-up question: ", convo_history_list_guest)
    return convo_history_list_guest


# function to get the past intention of the user
def get_past_conversation_guest(memory) :
    # Extract the last element from each tuple in the memory list
    if len(memory) == 0:
        return "", "", ""
    else: 
        # previous_intention, previous_follow_up_question, last_bot_response, previous_items_recommended
        print("line 106: ", memory)
        previous_intention = memory[-1][1]
        previous_follow_up_question = memory[-1][1].get("Follow-Up Question")
        previous_items_recommended = memory[-1][2]
        print("line 107: ", previous_follow_up_question) # data type: dictionary     
        
        return previous_intention, previous_follow_up_question, previous_items_recommended

def update_past_follow_up_question_guest(user_intention_dictionary):
    # function to update the follow up questions such that it can be passed on to the LLM during the next user prompt loop
    follow_up_question = user_intention_dictionary.get("Follow-Up Question")
    return follow_up_question

def get_past_conversations_to_display(user_id):
    # Retrieve all convo history for that user, sort it by timestamp
    all_sessions = db.chatSession.find(
        {"user_id": user_id},
        sort=[("created_at", pymongo.ASCENDING)]   
    )

    conversation_history = []
    
    # Loop through each session to retrieve messages
    for session in all_sessions:
        message_list = session.get("message_list", [])
        
        for message in message_list:
            user_input = message.get("user_input", "")
            follow_up = message.get("follow up", "").strip()

            # Format the recommendations provided by the bot since its not saved in the database
            items_recommended = message.get("items_recommended", [])
            if items_recommended:
                # Generate intro text dynamically based on product context
                intro_text = "Based on your past preferences, here are some recommendations:"
                recommended_items = "\n".join([
                    f"Product Name: {item.get('product_name', 'N/A')}, "
                    f"Brand: {item.get('brand', 'N/A')}, "
                    f"Price: ₹{item.get('retail_price', 'N/A')} "
                    f"(Discounted Price: ₹{item.get('discounted_price', 'N/A')} after {item.get('discount', 'N/A')}% discount)"
                    for item in items_recommended
                ])
                bot_message = f"{intro_text}\n\n{recommended_items}\n\n{follow_up}"
            else:
                # If no items are recommended, only include the follow-up question
                bot_message = follow_up

            # Add structured messages to the conversation history
            conversation_history.append({
                "user": user_input,
                "bot": bot_message
            })
    
    return conversation_history 