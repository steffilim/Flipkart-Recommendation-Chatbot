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

from convohistory import add_chat_history_guest, get_past_conversation_guest, get_past_conversations_users, add_chat_history_user, start_new_session
from prompt_template import intention_template, refine_template
from functions import is_valid_input, getting_bot_response, get_popular_items, getting_user_intention, initialising_mongoDB
from recSys.contentBased import get_lsa_matrix, load_product_data

# Initialize Flask app
app = Flask(__name__)

# Dummy user IDs for validation
valid_user_ids = ["U03589", "U08573", "U07482", "U07214", "U08218"]
keywords = ["/logout", "/login", "guest", "Guest"]
password = "pw123"  # Hardcoded password

# Initialisation
user_states = {}  # User states
query_steps = {}  # Track multi-step interaction state
convo_history_list_guest = []  # Conversation history for guest users
previous_intention = ""  # User intention
session_id = ""  # Session ID  

# INITIALISATION
# Authenticating model
load_dotenv()

def initialise_app():
    db = initialising_mongoDB()
    print("Database setup successful")

    # Load LSA matrix
    catalogue_df = load_product_data(db.catalogue)
    catalogue_db = db.catalogue
    lsa_matrix_file = 'lsa_matrix.joblib'
    lsa_matrix = get_lsa_matrix(catalogue_df, catalogue_db, lsa_matrix_file)

    google_api_key = os.getenv("GOOGLE_API_KEY")
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro", 
        google_api_key=google_api_key,
        temperature=0.1, 
        verbose=True, 
        stream=True
    )

    # CHAINING
    refine_prompt = ChatPromptTemplate.from_template(refine_template)
    chain2 =  refine_prompt | llm | StrOutputParser()

    intention_prompt = ChatPromptTemplate.from_template(intention_template)
    intention_chain = intention_prompt | llm | StrOutputParser()

    return db, llm, chain2, intention_chain, lsa_matrix

db, llm, chain2, intention_chain, lsa_matrix = initialise_app()

@app.route('/')
def index():
    # Check if the user is in guest mode from user_states
    guest_mode = user_states.get("guest_mode", False)
    user_id = user_states.get("user_id")

    # Determine what message to display on page load
    if guest_mode:
        welcome_message = "Welcome back! You are currently in Guest Mode. You may enter /login to exit Guest Mode."
    elif user_id:
        welcome_message = f"Welcome back! You are logged in as User ID: {user_id}. You may enter /logout to log out."
    else:
        welcome_message = "Welcome! Please enter your User ID or enter guest to enable guest mode."

    return render_template('index.html', welcome_message=welcome_message)

@app.route('/chat', methods=['POST'])
def chat():
    user_data = request.get_json()
    user_input = user_data.get('message')
    user_id = user_states.get("user_id")
    guest_mode = user_states.get("guest_mode", False)

    # Initialize multi-step interaction state for new sessions
    if user_id not in query_steps:
        query_steps[user_id] = {"step": 0, "initial_query": None, "brand": None, "specs": None}

    current_step = query_steps[user_id]["step"]

    # If user selects guest mode
    if not user_id and user_input.lower() == "guest":
        user_states["guest_mode"] = True
        query_steps[user_id]["step"] = 1
        query_steps[user_id]["initial_query"] = user_input
        return jsonify({'response': 'Welcome to Guest Mode. Let\'s get started! What brands are you interested in?'})

    # Guest mode handling
    if guest_mode:
        if user_input == "/login":
            user_states.pop("guest_mode", None)
            user_states["login_mode"] = True
            return jsonify({'response': 'Please enter your user ID to log in.'})

        # Check if the user input is product-related
        is_product_related = is_valid_input(user_input, valid_user_ids, keywords)

        if current_step == 0:
            if is_product_related:
                query_steps[user_id]["initial_query"] = user_input
                query_steps[user_id]["step"] = 1
                return jsonify({'response': 'What brands are you interested in?'})
            else:
                # Handle non-product-related inputs at step 0
                return jsonify({'response': "I'm here to assist with product recommendations. How can I help with your search today?"})

        elif current_step == 1:
            query_steps[user_id]["brand"] = user_input
            query_steps[user_id]["step"] = 2
            return jsonify({'response': 'Any specific specifications or features you are looking for?'})
        elif current_step == 2:
            query_steps[user_id]["specs"] = user_input
            query_steps[user_id]["step"] = 3

            # Extract brand and specs preferences
            initial_query = query_steps[user_id]["initial_query"]
            brand_preference = query_steps[user_id]["brand"]
            specs_preference = query_steps[user_id]["specs"]

            # Call getting_user_intention with separate parameters
            user_intention = getting_user_intention(
                initial_query,
                intention_chain,
                previous_intention,
                brand_preference=brand_preference,
                specs_preference=specs_preference
            )

            # Call getting_bot_response with separate parameters
            bot_response = getting_bot_response(
                user_intention,
                chain2,
                db,
                lsa_matrix,
                user_id=None,
                brand_preference=brand_preference,
                specs_preference=specs_preference
            )

            add_chat_history_guest(initial_query, bot_response, convo_history_list_guest)

            # Reset multi-step interaction for the next query
            query_steps[user_id] = {"step": 0, "initial_query": None, "brand": None, "specs": None}
            return jsonify({'response': bot_response})

    # Handle initial login or ID input
    if not user_id:
        if user_input in valid_user_ids:
            user_states["user_id"] = user_input
            user_states["password_mode"] = True
            user_states["session_id"] = str(uuid4())
            return jsonify({'response': 'User ID validated. Please enter your password.'})
        
        return jsonify({'response': 'Invalid ID. Please enter a valid numeric ID or type "guest" to continue without logging in.'})

    # Check if the user is prompted to enter a password
    if user_states.get("password_mode"):
        if user_input == password:
            user_states.pop("password_mode", None)
            return jsonify({'response': 'Password validated. How can I assist you with product recommendations today?'})
        else:
            return jsonify({'response': 'Incorrect password. Please try again.'})

    # Check if the user input is product-related
    is_product_related = is_valid_input(user_input, valid_user_ids, keywords)

    # Handle logged-in users
    previous_intention = get_past_conversations_users(user_id, user_states.get("session_id"))

    if current_step == 0:
        if is_product_related:
            query_steps[user_id]["initial_query"] = user_input
            query_steps[user_id]["step"] = 1
            return jsonify({'response': 'What brands are you interested in?'})
        else:
            # Handle non-product-related inputs at step 0
            return jsonify({'response': "I'm here to assist with product recommendations. How can I help with your search today?"})

    elif current_step == 1:
        query_steps[user_id]["brand"] = user_input
        query_steps[user_id]["step"] = 2
        return jsonify({'response': 'Any specific specifications or features you are looking for?'})
    elif current_step == 2:
        query_steps[user_id]["specs"] = user_input
        query_steps[user_id]["step"] = 3

        # Extract brand and specs preferences
        initial_query = query_steps[user_id]["initial_query"]
        brand_preference = query_steps[user_id]["brand"]
        specs_preference = query_steps[user_id]["specs"]

        # Call getting_user_intention with separate parameters
        user_intention = getting_user_intention(
            initial_query,
            intention_chain,
            previous_intention,
            brand_preference=brand_preference,
            specs_preference=specs_preference
        )

        # Call getting_bot_response with separate parameters
        bot_response = getting_bot_response(
            user_intention,
            chain2,
            db,
            lsa_matrix,
            user_id,
            brand_preference=brand_preference,
            specs_preference=specs_preference
        )

        add_chat_history_user(user_states.get("session_id"), initial_query, user_intention, bot_response)

        # Reset multi-step interaction for the next query
        query_steps[user_id] = {"step": 0, "initial_query": None, "brand": None, "specs": None}
        return jsonify({'response': bot_response})

    # Continue with normal bot response
    bot_response = getting_bot_response(
        user_intention,
        chain2,
        db,
        lsa_matrix,
        user_id,
        brand_preference=None,
        specs_preference=None
    )
    add_chat_history_user(user_states.get("session_id"), user_input, user_intention, bot_response)

    return jsonify({'response': bot_response})

if __name__ == '__main__':
    app.run(debug=True)