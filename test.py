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
from functions import is_valid_input, get_recommendation, extract_keywords,re_rank_with_intent


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
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    user_data = request.get_json()
    user_input = user_data.get('message')

    # Get user state to check if ID has already been provided
    user_id = user_states.get("user_id")
    session_id = user_states.get("session_id")

    # If user ID is not set, expect user to input the ID first
    
    if not user_id:
        try:
            user_id = int(user_input)
        except ValueError:
            return jsonify({'response': 'Invalid ID. Please enter a valid numeric user ID.'})

        if user_id in valid_user_ids:
            user_states["user_id"] = user_id  # Save the user ID
            user_states["session_id"] = str(uuid4())  # Generate a unique session ID; for each new convo, it should have a unique session ID 
            return jsonify({'response': 'User ID validated. Please enter your query.'})
        else:
            return jsonify({'response': 'Invalid ID. Please enter a valid user ID.'})

    # Now that user ID is validated, expect further prompts

    # Initialising a new session ID
    

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

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001, debug=True)
