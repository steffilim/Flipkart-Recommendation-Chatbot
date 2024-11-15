# current updated version of the gemini chatbot
import os
import pandas as pd

from uuid import uuid4 
from dotenv import load_dotenv
from collections import Counter
import re
import time
import json
import threading

# for LLM
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from flask import Flask, render_template, request, jsonify
from langchain_core.prompts import ChatPromptTemplate


from convohistory import add_chat_history_guest, get_past_conversation_guest, get_past_conversations_users, add_chat_history_user, start_new_session, update_past_follow_up_question_guest, get_past_conversations_to_display
from prompt_template import refine_template, intention_template
from functions import is_valid_input, getting_bot_response, get_popular_items, getting_user_intention_dictionary, initialising_mongoDB, extract_keywords, parse_user_intention, initialising_supabase, load_product_data, load_users_data, getting_user_purchase_dictionary, recommend_similar_products
#from recSys.contentBased import load_product_data





# Initialize Flask app
app = Flask(__name__)

# Dummy user IDs for validation
# valid_user_ids = ["U03589", "U08573", "U07482", "U07214", "U08218", "U01357"]
valid_user_ids = ["U01394", "U01357", "U01290", "U01385"]
keywords = ["/logout", "/login", "guest", "Guest"]
password = "pw123"  # Hardcoded password

# Initialisation
user_states = {} # user states
convo_history_list_guest = [] # convo history list for guest users
previous_intention = "" # user intention
session_id = "" # session id    
past_follow_up_question_guest = "" # follow up question for guest users


# INITIALISATION
# Authenticating model
load_dotenv()






# initialising memory

# Flask routes

def initialise_app():
    supabase = initialising_supabase()
    print("Supabase setup successful")

    db = initialising_mongoDB()
    print("Database setup successful")

    google_api_key = os.getenv("GOOGLE_API_KEY")
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro", 
        google_api_key=google_api_key,
        temperature=0, 
        verbose=True, 
        stream=True
    )
    lsa_matrix= 'lsa_matrix.joblib'

    # CHAINING
    # refining the output based on the recommendations and keywords 
    refine_prompt = ChatPromptTemplate.from_template(refine_template)
    chain2 =  refine_prompt | llm | StrOutputParser()

    # Create a new chain for intention extraction
    intention_prompt = ChatPromptTemplate.from_template(intention_template)
    intention_chain = intention_prompt | llm | StrOutputParser()

    convo_history_list_guest = [] # convo history list for guest users


    return supabase, db, llm, chain2, intention_chain, lsa_matrix, convo_history_list_guest

supabase, db, llm, chain2, intention_chain, lsa_matrix, convo_history_list_guest = initialise_app()



@app.route('/')
def index():
    user_id = user_states.get("user_id")  # Check if user is logged in
    guest_mode = user_states.get("guest_mode")  # Check if guest mode is active

    # Determine what message to display on page load
    if user_states:
        if guest_mode:
            welcome_message = "Welcome back! You are currently in Guest Mode. You may enter /login to exit Guest Mode."
        else:
            welcome_message = f"Welcome back! You are logged in as User ID: {user_id}. You may enter /logout to log out."
    else:
        welcome_message = "Welcome! Please enter your User ID or enter guest to enable guest mode."

    return render_template('index.html', welcome_message=welcome_message)

@app.route('/chat', methods=['POST'])
def chat():

    user_data = request.get_json()
    user_input = user_data.get('message')
    user_id = user_states.get("user_id")
    session_id = user_states.get("session_id")

    # Check if the user is in login mode and expects a user ID input
    if user_states.get("login_mode"):
        try:
            user_id = str(user_input)

        except ValueError:
            return jsonify({'response': 'Invalid ID. Please enter a valid numeric user ID.'})

        if user_id in valid_user_ids:
            user_states["user_id"] = user_id
            user_states["password_mode"] = True  # Set password mode flag
            user_states.pop("login_mode", None)  # Remove login mode flag
            return jsonify({'response': 'User ID validated. Please enter your password.'})
        else:
            return jsonify({'response': 'Invalid ID. Please enter a valid numeric user ID.'})

    # If the user is prompted to enter a password
    if user_states.get("password_mode"):
        if user_input == "pw123":  # Hardcoded password check
              # Start a new session for the user
            print("line 132")

            user_states["session_id"] = str(uuid4())  # Generate a unique session ID            
            session_id = user_states["session_id"]

            # Retrieve the past conversation messages 
            past_conversations = get_past_conversations_to_display(user_states["user_id"])
            # If user dont have any past conversations, clear the chat and start fresh
            if not past_conversations:
                start_new_session(user_id, session_id)            
                user_states.pop("password_mode", None)  # Remove password mode flag
                user_id = user_states["user_id"]

                recommendations = recommend_similar_products(user_id)
                response_text = "Welcome back! Here are some products you might be interested in:\n" + "\n".join(recommendations)
                response_text += "\n\nWould you like to know more about any of these items? If not, please provide me the description of the item you are looking for. You may enter /logout to log out."
                
                # Clear chat and display the default login message
                return jsonify({
                    'clear_chat': True,  
                    'response': response_text
                })
            else:
                # Continue with past conversation handling
                start_new_session(user_id, session_id)
                user_states.pop("password_mode", None)
                user_id = user_states["user_id"]

                recommendations = recommend_similar_products(user_id)
                response_text = "Welcome back! Here are some products you might be interested in:\n" + "\n".join(recommendations)
                response_text += "\n\nWould you like to know more about any of these items? If not, please provide me the description of the item you are looking for. You may enter /logout to log out."
                return jsonify({
                    'past_conversations': past_conversations,
                    'response': response_text
                })
        else:
            return jsonify({'response': 'Incorrect password. Please try again.'})

    # Validate user input only when not in login or password mode

    # Check if the user is logged in and wants to log out
    if user_states.get("user_id") and user_input == "/logout":
        user_states.pop("user_id", None)  # Remove user ID
        user_states.pop("session_id", None)  # Remove session ID
        user_states["guest_mode"] = True  # Set guest mode flag

        return jsonify({'response': 'You have logged out and are now in guest mode. You may enter /login to log in again.'})

    # Check if the user is in guest mode
    if user_states.get("guest_mode"):
        past_follow_up_question_guest = ""
     

        if user_input == "/login":
            user_states.pop("guest_mode", None)  # Remove guest mode flag
            user_states["login_mode"] = True  # Set login mode flag

            return jsonify({'response': 'Please enter your user ID to log in.'})

        # Get previous conversation and intention in guest mode
        if not is_valid_input(user_input, valid_user_ids, keywords):
            return jsonify({'response': "I'm sorry, I do not understand what you meant. Please rephrase or ask about a product available in our store."})

        previous_intention, previous_follow_up_question, previous_items_recommended = get_past_conversation_guest(convo_history_list_guest)
        print("previous_items: ", previous_items_recommended)
        user_intention = getting_user_intention_dictionary(user_input, intention_chain, previous_intention, past_follow_up_question_guest, previous_items_recommended)
        user_intention_dictionary = parse_user_intention(user_intention)
        #print("line 172:", user_intention_dictionary)

        # updating follow-up question
        past_follow_up_question_guest = update_past_follow_up_question_guest(user_intention_dictionary)
        recommendations, bot_response = getting_bot_response(user_intention_dictionary, chain2, supabase, db, previous_items_recommended, user_profile = "None", user_purchases = "None", user_id = "guest", session_id = "guest")
        
        # To account for the case of the user asking for more information on a particular item
        if recommendations is None:
            recommendations = previous_items_recommended
        
        add_chat_history_guest(user_input, user_intention_dictionary, recommendations, convo_history_list_guest)
        print("line 189:, ", convo_history_list_guest)


        return jsonify({'response': bot_response}) 
    

    

    # Get user state to check if ID has already been provided
    
    user_profile = []
    user_purchases = []

   # If the user is prompted to enter user ID (after /login)
    if user_states.get("login_mode"):
        try:
            # Try to interpret the input as an ID
            user_id = str(user_input)
        except ValueError:
            # If input is not a valid numeric ID, prompt for valid ID again
            return jsonify({'response': 'Invalid ID. Please enter a valid numeric user ID.'})

        if user_id in valid_user_ids:
            # Valid user ID, store it and initialize a session
            user_states["user_id"] = user_id  # Save the user ID
            user_states.pop("login_mode", None)  # Remove login mode flag
            return jsonify({'response': 'User ID validated. You may enter /logout to exit. Please enter your query.'})
        else:
            return jsonify({'response': 'Invalid ID. Please enter a valid numeric user ID.'})


    # Get user state to check if ID has already been provided
    user_id = user_states.get("user_id")
    session_id = user_states.get("session_id")

    # If user ID is not set, expect user to input the ID or choose guest mode
    if not user_id:
        popular_items_recommendation = get_popular_items()
        if user_input.lower() == "guest":

            user_states["guest_mode"] = True  # Set guest mode flag
            return jsonify({'response': popular_items_recommendation})

        try:

            # Try to interpret the input as an ID
            user_id = str(user_input)

        except ValueError:
            return jsonify({'response': 'Invalid ID. Please enter a valid numeric ID, or type "guest" to continue without logging in.'})

        if user_input in valid_user_ids:

                # Valid user ID, store it and initialize a session
            user_states["user_id"] = user_id  # Save the user ID
            user_states.pop("guest_mode", None)  # Ensure guest mode flag is removed
            user_states["password_mode"] = True  # Set password mode flag
            
            return jsonify({'response': 'User ID validated. Please enter your password.'})

        else:
            return jsonify({'response': 'Invalid ID. Please enter a valid numeric ID, or type "guest" to continue without logging in.'})

    # Getting information from the user
    if not is_valid_input(user_input, valid_user_ids, keywords):
        return jsonify({'response': "I'm sorry, I do not understand what you meant. Please rephrase or ask about a product available in our store."})
  
    previous_intention, previous_follow_up_question, previous_items_recommended = get_past_conversations_users(user_id, session_id)    
    user_intention = getting_user_intention_dictionary(user_input, intention_chain, previous_intention, previous_follow_up_question, previous_items_recommended)
    user_intention_dictionary = parse_user_intention(user_intention)
    print(user_intention_dictionary)


    # Getting the bot response
    recommendations, bot_response = getting_bot_response(user_intention_dictionary, chain2, supabase, db, previous_items_recommended, user_profile, user_purchases, user_id, session_id)

    # To account for the case of the user asking for more information on a particular item
    if recommendations is None:
        recommendations = previous_items_recommended
        

    print(session_id)
    add_chat_history_user(session_id, user_input, user_intention_dictionary, recommendations)

    print("Chat history updated successfully")
    
    return jsonify({'response': bot_response})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001, debug=True)