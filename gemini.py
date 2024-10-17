# current updated version of the gemini chatbot
import os
import pandas as pd

from uuid import uuid4 
from dotenv import load_dotenv
from collections import Counter
import re
import time

# for LLM
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from flask import Flask, render_template, request, jsonify
from langchain_core.prompts import ChatPromptTemplate





from convohistory import add_chat_history_user, get_past_conversations_users, add_chat_history_guest, get_past_conversation_guest
from prompt_template import intention_template, refine_template
from functions import is_valid_input, getting_bot_response, get_popular_items, getting_user_intention




# Initialize Flask app
app = Flask(__name__)

# Dummy user IDs for validation
valid_user_ids = [78126, 65710, 58029, 48007, 158347]
keywords = ["/logout", "/login", "guest", "Guest"]

# Initialisation
user_states = {} # user states
convo_history_list_guest = [] # convo history list for guest users
previous_intention = "" # user intention

# INITIALISATION
# Authenticating model
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(
        model="gemini-pro", 
        google_api_key=google_api_key,
        temperature=0.1, 
        verbose=True, 
        stream=True
    )


# CHAINING

# refining the output based on the recommendations and keywords 
refine_template = ChatPromptTemplate.from_template(refine_template)
chain2 =  refine_template | llm | StrOutputParser()

# Create a new chain for intention extraction
intention_prompt = ChatPromptTemplate.from_template(intention_template)
intention_chain = intention_prompt | llm | StrOutputParser()

# initialising memory

# Flask routes
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

    if not is_valid_input(user_input, valid_user_ids, keywords):
        return jsonify({'response': "I'm sorry, I do not understand what you meant. Please rephrase or ask about a product available in our store."})


    # Check if the user is logged in and wants to log out
    if user_states.get("user_id") and user_input == "/logout":
        # Log the user out and switch to guest mode
        user_states.pop("user_id", None)  # Remove user ID
        user_states.pop("session_id", None)  # Remove session ID
        user_states["guest_mode"] = True  # Set guest mode flag
        
        return jsonify({'response': 'You have logged out and are now in guest mode. You may enter /login to log in again.'})

    # To check if the user is in guest mode
    if user_states.get("guest_mode"):
        if user_input == "/login":
            # If the user enters "/login", switch from guest mode to login prompt
            user_states.pop("guest_mode", None)  # Remove guest mode flag
            user_states["login_mode"] = True     # Set login mode flag
            return jsonify({'response': 'Please enter your user ID to log in.'})
        
        previous_intention = get_past_conversation_guest(convo_history_list_guest)
        user_intention = getting_user_intention(user_input, intention_chain, previous_intention)

        bot_response = getting_bot_response(user_intention, chain2)
        add_chat_history_guest(user_input, bot_response, convo_history_list_guest)
        print(convo_history_list_guest)

        return jsonify({'response': bot_response}) # TO BE UPDATED WITH STREAMLINE CODE
    
    # If the user is prompted to enter user ID (after /login)
    if user_states.get("login_mode"):
        try:
            # Try to interpret the input as an ID
            user_id = int(user_input)
        except ValueError:
            # If input is not a valid numeric ID, prompt for valid ID again
            return jsonify({'response': 'Invalid ID. Please enter a valid numeric user ID.'})

        if user_id in valid_user_ids:
            # Valid user ID, store it and initialize a session
            user_states["user_id"] = user_id  # Save the user ID
            user_states["session_id"] = str(uuid4())  # Generate a unique session ID
            user_states.pop("login_mode", None)  # Remove login mode flag
            return jsonify({'response': 'User ID validated. You may enter /logout to exit. Please enter your query.'})
        else:
            return jsonify({'response': 'Invalid ID. Please enter a valid numeric user ID.'})
        
    # Get user state to check if ID has already been provided
    user_id = user_states.get("user_id")

    # If user ID is not set, expect user to input the ID or choose guest mode
    if not user_id:
        popular_items_recommendation = get_popular_items()
        if user_input == "guest" or user_input == "Guest":
            # If the user opts for guest mode
            user_states["guest_mode"] = True  # Set guest mode flag

            return jsonify({'response': popular_items_recommendation})

        try:
            # Try to interpret the input as an ID
            user_id = int(user_input)
        except ValueError:
            # If input is not a valid numeric ID, proceed in guest mode
            return jsonify({'response': 'Invalid ID. Please enter a valid numeric ID, or type "guest" to continue without logging in.'})

        if user_id in valid_user_ids:
            # Valid user ID, store it and initialize a session
            user_states["user_id"] = user_id  # Save the user ID
            user_states["session_id"] = str(uuid4())  # Generate a unique session ID
            user_states.pop("guest_mode", None)  # Ensure guest mode flag is removed
            return jsonify({'response': 'User ID validated. You may enter /logout to exit. Please enter your query.'})
        else:
            return jsonify({'response': 'Invalid ID. Please enter a valid numeric ID, or type "guest" to continue without logging in.'})

    # Getting information from the user
    previous_intention = get_past_conversations_users(user_id, user_states["session_id"])
    user_intention = getting_user_intention(user_input, intention_chain, previous_intention)

    # Getting the bot response
    bot_response = getting_bot_response(user_intention, chain2)
    add_chat_history_user(user_id, user_states["session_id"], user_input, bot_response, user_intention)
    
    return jsonify({'response': bot_response})


if __name__ == '__main__':
    app.run(debug=True)