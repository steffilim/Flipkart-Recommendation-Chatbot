# baseline code for the chatbot
import os
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import SequentialChain, LLMChain
import pandas as pd
from collections import Counter

# Import add_chat_history function
from convohistory import add_chat_history, get_past_conversations
from prompt_template import intention_template, keywords_template, refine_template

# Initialize Flask app
app = Flask(__name__)

# Dummy user IDs for validation
valid_user_ids = [78126, 65710, 58029, 48007, 158347]

# To store the user state (whether the user has provided a valid ID)
user_states = {}

# INITIALISATION
# Authenticating model
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY2")
llm = ChatGoogleGenerativeAI(
        model="gemini-pro", 
        google_api_key=google_api_key,
        temperature=0.2, 
        verbose=True, 
        stream=True
    )

# Initializing data
catalouge = pd.read_csv('newData/flipkart_cleaned.csv')
purchase_history = pd.read_csv('newData/synthetic_v2.csv')

 

# Create a new chain for intention extraction
intention_prompt = PromptTemplate(input_variables=["user_input"], template=intention_template)
intention_chain = LLMChain(llm=llm, output_key="intention", prompt=intention_prompt)


# Creating a sample recommender system
def get_recommendation(keywords_list): # getting the top 3 products based on keywords
    mask = catalouge['product_category_tree'].apply(lambda x: any(keyword in x for keyword in keywords_list))
    filtered = catalouge[mask]
    top_products = filtered.sort_values(by='overall_rating', ascending=False).head(3)

    # Formatting the output more clearly
    return "\n".join(
        f"**{idx + 1}. {row['product_name']}** - Discounted Price: {row['discounted_price']}, Description: {row['description']}"
        for idx, row in top_products.iterrows()
    )


# CHAINING
# Chaining the recommendation system
promt_template_1 = PromptTemplate(input_variables=["question"], template=keywords_template, verbose=True)
chain1 = LLMChain(llm=llm, output_key="query", prompt=promt_template_1)

prompt_template_2 = PromptTemplate(input_variables=["query"], template=refine_template)
chain2 = LLMChain(llm=llm, output_key="refined", prompt=prompt_template_2, verbose=True)

ssChain = SequentialChain(chains=[chain1, chain2],
                          input_variables=["question"],
                          output_variables=["refined"],
                          verbose=True)


def to_list(text):
    return text.split(',')

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

    # If user ID is not set, expect user to input the ID first
    if not user_id:
        try:
            user_id = int(user_input)
        except ValueError:
            return jsonify({'response': 'Invalid ID. Please enter a valid numeric user ID.'})

        if user_id in valid_user_ids:
            user_states["user_id"] = user_id  # Save the user ID
            return jsonify({'response': 'User ID validated. Please enter your query.'})
        else:
            return jsonify({'response': 'Invalid ID. Please enter a valid user ID.'})

    # Now that user ID is validated, expect further prompts
    # Extract intention from the user input
    intention_result = intention_chain.invoke(input=user_input)
    user_intention = intention_result['intention']


    intermediate_results = chain1.invoke(input=user_input)
    results_ls = to_list(intermediate_results['query'])
    print("Results list: ", results_ls)

    

    if results_ls[0] == "Greeting":
        bot_response = "Hello! How can I help you today?"

    elif results_ls[0] == "None":
        #result = ssChain.invoke(input=user_input)
        #bot_response = result['refined']
        bot_response = "I'm sorry, I'm not able to help you with that. Would you like to search for something else? If not, please try searching for something else or contact customer service for assistance."

    else:
        # Get recommendations based on user's purchase history and extracted keywords
        print("Getting recommendations")
        recommendations = get_recommendation(results_ls)
        print("Recommendations: ", recommendations)
        result = chain2.invoke(input=recommendations)
        print(result)
        bot_response = result['refined']

    # Call the add_chat_history function to save the convo
    add_chat_history(user_id, user_input, bot_response, user_intention)

    # getting past conversation history from the database. 
    print(get_past_conversations(user_id)) 

    return jsonify({'response': bot_response})



if __name__ == '__main__':
    app.run(debug=True)
