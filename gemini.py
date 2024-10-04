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

# Initialize Flask app
app = Flask(__name__)

# Dummy user IDs for validation
valid_user_ids = [78126, 65710, 58029, 48007, 158347]

# To store the user state (whether the user has provided a valid ID)
user_states = {}

# LLM INITIALISATION
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

 
intention_template = """
You are a smart e-commerce chatbot. Based on the user's input, classify the intent with more specificity, identifying the context and product type if possible.

Here is the input that you have received: {user_input}

For example:
- If the user is looking for a product recommendation for a specific need (e.g., school, gaming, shirts), classify it as "Product recommendation for [specific need]."
- If the user is inquiring about product features, classify it as "Product feature inquiry for [specific product]."
- If the user is asking for customer support, classify it as "Customer support request."
Return a short phrase summarizing the intent.
"""

intention_prompt = PromptTemplate(input_variables=["user_input"], template=intention_template)
# Create a new chain for intention extraction
intention_chain = LLMChain(llm=llm, output_key="intention", prompt=intention_prompt)


template1 = """
You are a smart LLM for an e-commerce company.
Imagine that you are selling the same products as Amazon.com.
You are to identify the keywords in the query of the user. 
Here is the input that you have received {question}.

If the user is asking for a product that cannot be found in the e-commerce company, the output for the list should be "None".

Else If the user is doing a greeting, the output for the list should be "Greeting".
Else If the user is referring back to its 


Else, identify the important keywords in the query and put them in a list and separate the keywords with a comma.
"""

template2 = """
You are a refined recommendation engine chatbot for an e-commerce online company.
Your job is to refine the output based off the input that has given to you. 

You have received a list from the previous LLM with the following keywords: {query}.

It contains the following headers: `Product Name`, `Price`, `Description`.
Extract the relevant information from the list and provide a response that is clear to the user. 

Summarise the product description. 
Omit the product number and give it in the following format:
For each product, follow the following format with markdown bold containers (**) for each product:
Product Name: <product_name>  
Price: <price>  
<description>
You should ask the user if the provided recommendations suit their needs or if they want another set of recommendations. 


"""

# DATA INITIALISATION
# Initializing data
catalouge = pd.read_csv('newData/flipkart_cleaned.csv')
purchase_history = pd.read_csv('newData/synthetic_v2.csv')

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
promptTemplate1 = PromptTemplate(input_variables=["question"], template=template1, verbose=True)
chain1 = LLMChain(llm=llm, output_key="query", prompt=promptTemplate1)

promptTemplate2 = PromptTemplate(input_variables=["query"], template=template2)
chain2 = LLMChain(llm=llm, output_key="refined", prompt=promptTemplate2, verbose=True)

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
    
    return jsonify({'response': bot_response})



if __name__ == '__main__':
    app.run(debug=True)
