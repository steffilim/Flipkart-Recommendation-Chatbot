# current updated version of the gemini chatbot
import os

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



from convohistory import add_chat_history, get_past_conversations
from prompt_template import intention_template, refine_template
from functions import is_valid_input, get_recommendation, extract_keywords




# Initialize Flask app
app = Flask(__name__)

# Dummy user IDs for validation
valid_user_ids = [78126, 65710, 58029, 48007, 158347]

# To store the user state (whether the user has provided a valid ID)
user_states = {}

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
        return handle_guest_mode(user_input)
    
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
        if user_input == "guest" or user_input == "Guest":
            # If the user opts for guest mode
            user_states["guest_mode"] = True  # Set guest mode flag
            return jsonify({'response': 'You are in guest mode now! You may enter /login to exit guest mode. What would you like to enquire?'})

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

    # Now that user ID is validated, expect further prompts
    # Initialising a new session ID
    session_id = user_states.get("session_id")

    # Check if the user input is valid
    if not is_valid_input(user_input):
        return jsonify({'response': "I'm sorry, I do not understand what you meant. Please rephrase or ask about a product available in our store."})

    # Getting past conversation history 
    user_convo_history = get_past_conversations(user_id, session_id)
    user_convo_history_string = " ".join(d['intention'] for d in user_convo_history)
    print("User convo history: ", user_convo_history_string)

    previous_intention = ""
    if user_convo_history_string != "":
        previous_intention_match = re.search(r'Actionable Goal \+ Specific Details: ([^.\n]+)', user_convo_history_string)
        previous_intention = previous_intention_match.group(1) 
        print("Previous intention:", previous_intention)

    # Get the user current intention
    user_intention = intention_chain.invoke({"input": user_input, "previous_intention": previous_intention})
    print("User intention: ", user_intention)

    # Getting item status
    match = re.search(r'Available in Store:\s*(.+)', user_intention)
    available_in_store = match.group(1)

    if available_in_store != "Yes." :
        # Getting suggested response/ follow up action if item is not found in the store
        response = re.search(r'Suggested Actions or Follow-Up Questions:\s*(.+)', user_intention, re.DOTALL)
        bot_response = response.group(1).strip()
    else:
        # Getting item of interest
        match = re.search(r'Actionable Goal \+ Specific Details:\s*(.+)', user_intention)
        item = match.group(1)
        #start_time = time.time()
        # Getting recommendations from available products
        query_keyword_ls = extract_keywords(item)
        print("keywords: ", query_keyword_ls)
        #print("Time taken: ", time.time() - start_time)

        # Getting the follow-up questions from the previous LLM
        questions_match = re.search(r'Suggested Actions or Follow-Up Questions:\s*(.+)', user_intention, re.DOTALL)
        questions = questions_match.group(1).strip()
        recommendations = get_recommendation(query_keyword_ls)
        bot_response = chain2.invoke({"recommendations": recommendations, "questions": questions})


    # Call the add_chat_history function to save the convo
    add_chat_history(user_id, session_id, user_input, bot_response, user_intention)
    
    return jsonify({'response': bot_response})

def handle_guest_mode(user_input):
    """Handle guest mode where no user ID is stored or conversation history saved."""
    print("Guest mode activated.")
    
    # Process the user input like a normal conversation without storing history
    if not is_valid_input(user_input):
        return jsonify({'response': "I'm sorry, I do not understand what you meant. Please rephrase or ask about a product available in our store."})

    # Get the user intention without storing any session info
    # Set previous_intention as an empty string for guest mode
    user_intention = intention_chain.invoke({"input": user_input, "previous_intention": ""})
    print("Guest user intention: ", user_intention)

    # Getting item status
    match = re.search(r'Available in Store:\s*(.+)', user_intention)
    available_in_store = match.group(1)

    if available_in_store != "Yes.":
        # Getting suggested response/follow-up action if the item is not found in the store
        response = re.search(r'Suggested Actions or Follow-Up Questions:\s*(.+)', user_intention, re.DOTALL)
        bot_response = response.group(1).strip()
    else:
        # Getting item of interest
        match = re.search(r'Actionable Goal \+ Specific Details:\s*(.+)', user_intention)
        item = match.group(1)

        # Getting recommendations from available products
        query_keyword_ls = extract_keywords(item)
        print("keywords: ", query_keyword_ls)

        # Getting the follow-up questions from the previous LLM
        questions_match = re.search(r'Suggested Actions or Follow-Up Questions:\s*(.+)', user_intention, re.DOTALL)
        questions = questions_match.group(1).strip()
        recommendations = get_recommendation(query_keyword_ls)
        bot_response = chain2.invoke({"recommendations": recommendations, "questions": questions})

    # Respond to guest user without saving anything
    return jsonify({'response': bot_response})

if __name__ == '__main__':
    app.run(debug=True)