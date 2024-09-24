import os
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import SequentialChain, LLMChain
import pandas as pd
from collections import Counter

# LLM INITIALISATION
# authenticating model
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(
        model="gemini-pro", 
        google_api_key=google_api_key,
        temperature=0.2, 
        verbose=True, 
        stream=True
    )

# Prompt Templates
template1 = """
You are a smart LLM for an e-commerce company.
Imagine that you are selling the same products as Amazon.com.
You are to identify the keywords in the query of the user. 
Here is the input that you have received {question}.

If the user is asking for a product that cannot be found in the e-commerce company, you should tell them that you are not able to help and should either search for something else or ask for customer service. 
The output for the list should be "None".
Else, identify the important keywords in the query and put them in a list and separate the keywords with a comma.
"""

template2 = """
You are a refined recommendation engine chatbot for an e-commerce online company.
Your job is to refine the output based on the input given to you. 

If you have received a list with the word "None", you should tell the user that you are not able to help and should either search for something else or ask for customer service.

Else:
You have been given a list of recommendations with the following headers: Product Name, Price, Description, User Interests.
Make sure the recommendations align with the user's interests, if possible. 
Summarise the product description. 
Omit the product number and give it in the following format:
1. **Product Name**: Price, 
 - Description

Here is the user's question: {query}
You should ask the user if the provided recommendations suit their needs or if they want another set of recommendations.
"""

# Data Initialisation
catalouge = pd.read_csv('newData/flipkart_cleaned.csv')
purchase_history = pd.read_csv('newData/synthetic_v2.csv')

# Creating a sample recommender system
def get_recommendation(user_id, keywords_list):
    # Step 1: Filter catalog based on keywords to create List A
    keyword_mask = catalouge['product_category_tree'].apply(lambda x: any(keyword in x for keyword in keywords_list))
    list_A = catalouge[keyword_mask]
    
    # Step 2: Get the user's purchase history
    user_purchases = purchase_history[purchase_history['User ID'] == user_id]

    # Step 3: Extract the product category tree from previous purchases and tokenize words
    purchased_categories = user_purchases['Product ID'].map(
        lambda x: catalouge[catalouge['uniq_id'] == x]['product_category_tree'].values[0]
    )

    # Tokenize the product category tree into individual words and count frequencies
    category_words = Counter(word.strip(" '") for category_tree in purchased_categories for word in eval(category_tree))

    # Get the top 5 most common terms from the user's purchase history
    most_frequent_words = [word for word, _ in category_words.most_common(5)]  # Top 5 most frequent terms

    # Step 4: From List A, filter the items based on most frequent words in their product category trees
    def match_category_tree(product_category_tree):
        return sum(1 for word in most_frequent_words if word in product_category_tree)

    list_A['matches'] = list_A['product_category_tree'].apply(lambda x: match_category_tree(eval(x)))

    # Filter by number of matches and then sort by matches and overall rating
    filtered = list_A[list_A['matches'] > 0].sort_values(by=['matches', 'overall_rating'], ascending=[False, False])

    # Step 5: Return the top 3 products
    top_products = filtered.head(3)

    # Format the output for the user
    return "\n".join(
        f"**{idx + 1}. {row['product_name']}** - Discounted Price: {row['discounted_price']}, Description: {row['description']}"
        for idx, row in top_products.iterrows()
    )

# LangChain Chains
promptTemplate1 = PromptTemplate(input_variables=["question"], template=template1)
chain1 = LLMChain(llm=llm, output_key="query", prompt=promptTemplate1)

promptTemplate2 = PromptTemplate(input_variables=["query"], template=template2)
chain2 = LLMChain(llm=llm, output_key="refined", prompt=promptTemplate2, verbose=True)

ssChain = SequentialChain(chains=[chain1, chain2], input_variables=["question"], output_variables=["refined"], verbose=True)

def to_list(text):
    return text.split(',')

# Flask App Initialisation
app = Flask(__name__)

while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    intermediate_results = chain1.invoke(input=prompt)
    results_ls = to_list(intermediate_results['query'])
    print(results_ls)
    
    if len(results_ls) <= 1:  # no recommendations found
        result = ssChain.invoke(input=prompt)
        print(result['refined'])       
        # Store conversation history
        add_chat_history(prompt, result['refined'])  # Call the function to add convo history into database

    if len(results_ls) <= 1: # no recommendations found
        print("NEED MORE INFORMATION")
        result = ssChain.invoke(input = prompt)
        print(result['refined'])
        continue
    else:
        # Get recommendations based on user's purchase history and the same product category tree
        recommendations = get_recommendation(user_id, results_ls)  # Pass user_id and extracted keywords to get_recommendation
        # Invoke the second chain for refining recommendations
        result = chain2.invoke(input=recommendations)
        print(result['refined'])

        # Assuming 'result' is a dictionary returned with refined recommendations
        #print(result['query'])
        #print(intermediate_results['text'])
        continue


        add_chat_history(prompt, result['refined'])  # Call the function to add convo history into database
        continue
        
# Route to serve the frontend (HTML)
@app.route('/')
def index():
    return render_template('index.html')

# API route to handle chatbot interaction
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_id = data.get('user_id')
    user_input = data.get('message')

    # Process user input through the first chain (keywords extraction)
    intermediate_results = ssChain.invoke(input=user_input)
    results_ls = to_list(intermediate_results['query'])

    if len(results_ls) <= 1:  # No recommendations found
        result = ssChain.invoke(input=user_input)
        response = result['refined']
    else:
        # Get recommendations based on user's purchase history and keywords
        recommendations = get_recommendation(user_id, results_ls)
        result = ssChain.invoke(input=recommendations)
        response = result['refined']

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)