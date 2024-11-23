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
    """
    Starts a new chat session in the database for a given user.

    Args:
        user_id (str): The unique identifier for the user.
        session_id (str): The unique identifier for the session.

    Returns:
        None
    """

    document = {
        "user_id": user_id,
        "session_id": session_id,
        "created_at": datetime.datetime.now(),
        "message_list": []
    }
    chat_session.insert_one(document)
    
def add_chat_history_user(session_id, user_input, user_intention_dictionary, items_recommended):
    """
    Adds the chat history of a registered user to the database.

    Args:
        session_id (str): The session identifier to find the corresponding session.
        user_input (str): The user's message/input.
        user_intention_dictionary (dict): The extracted user intentions.
        items_recommended (list): List of items recommended based on user input.

    Returns:
        None
    """

    # Build the dictionary to be pushed into the message list
    follow_up = user_intention_dictionary.get("Follow-Up Question")
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
    """
    Retrieves past conversations for a registered user from the database.

    Args:
        user_id (str): The unique identifier of the user.
        session_id (str): The session identifier to retrieve the conversation.

    Returns:
        Tuple: A tuple containing the previous user's intention, follow-up question, and recommended items.
    """

    initialisation  = chat_session.find_one({"user_id": user_id, "session_id": session_id})
    if initialisation is None:
        return "", "", ""
    
    # getting the past user inputs
    past_chats = initialisation["message_list"]
    if len(past_chats) <= 0:
        return "", "", ""
    
    else:
        previous_intention = past_chats[-1]["user_intention"]
        previous_follow_up_question = past_chats[-1]["follow up"]
        previous_items_recommended = past_chats[-1]["items_recommended"]
   

        return previous_intention, previous_follow_up_question, previous_items_recommended

"""CONVERSATION HISTORY FOR GUEST USERS """
# function to update the chat history (which is stored in a list)
def add_chat_history_guest(user_input, user_intention_dictionary, recommendations, convo_history_list_guest):
    """
    Adds the chat history for a guest user to the in-memory conversation history list.

    Args:
        user_input (str): The user's input/query.
        user_intention_dictionary (dict): The extracted user intentions.
        recommendations (list): List of items recommended.
        convo_history_list_guest (list): The in-memory list of conversation history for guest users.

    Returns:
        list: Updated conversation history list.
    """

    # Append the current user query and bot response as a tuple to the conversation history
    if user_intention_dictionary.get("Related to Follow-Up Questions") == "Old":
        follow_up_question = user_intention_dictionary.get("Follow-Up Question")
        convo_history_list_guest.append((user_input, user_intention_dictionary, recommendations, follow_up_question))
        convo_history_list_guest = convo_history_list_guest[1:] # removing the last element from the list
    else:
        convo_history_list_guest.append((user_input, user_intention_dictionary, recommendations))

    return convo_history_list_guest

def get_past_conversation_guest(memory):
    """
    Retrieves the past conversation details for a guest user from the in-memory conversation history.

    Args:
        memory (list): The in-memory conversation history list for a guest user.

    Returns:
        Tuple: A tuple containing the previous user's intention, follow-up question, and recommended items.
    """

    # Extract the last element from each tuple in the memory list
    if len(memory) == 0:
        return "", "", ""
    else: 
        previous_intention = memory[-1][1]
        previous_follow_up_question = memory[-1][1].get("Follow-Up Question")
        previous_items_recommended = memory[-1][2]
        
        return previous_intention, previous_follow_up_question, previous_items_recommended

def update_past_follow_up_question_guest(user_intention_dictionary):
    """
    Updates and retrieves the follow-up question for a guest user.

    Args:
        user_intention_dictionary (dict): The extracted user intentions.

    Returns:
        str: The follow-up question extracted from the dictionary.
    """

    # function to update the follow up questions such that it can be passed on to the LLM during the next user prompt loop
    follow_up_question = user_intention_dictionary.get("Follow-Up Question")

    return follow_up_question

def get_past_conversations_to_display(user_id):
    """
    Retrieves all past conversations for a registered user, sorted by timestamp.

    Args:
        user_id (str): The unique identifier for the user.

    Returns:
        list: A list of dictionaries containing the user input and bot responses formatted for display.
    """
    
    # Retrieve all convo history for that user, sort it by timestamp
    all_sessions = db.chatSession.find(
        {"user_id": user_id},
        sort=[("created_at", pymongo.ASCENDING)]   
    )

    conversation_history = []
    
    # Loop through each session to retrieve messages
    for session in all_sessions:
        message_list = session.get("message_list", [])
        if message_list:
            created_at = session["created_at"]
            #session_date = f"Session started on {created_at.strftime('%d %B %Y, %-I:%M %p').lower()} ---"
            session_date = session["created_at"].strftime("--- Session started on %d %B %Y, %H:%M ---")
            session_id = session["session_id"]

            conversation_history.append({
                "session_id": session_id,
                "session_date": session_date,
                "user": None,
                "bot": None
            })
                 
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
                "session_id": session_id,
                "user": user_input,
                "bot": bot_message
            })
    
    return conversation_history 