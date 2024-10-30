""" DATABASE FUNCTION"""

# Initializing data
import pandas as pd
from pymongo import MongoClient
import pymongo
import os
import currency
from dotenv import load_dotenv

INR = currency.symbol('INR')


def initialising_mongoDB():
    load_dotenv()
    MONGODB_URI = os.getenv("MONGODB_URI")
    FLIPKART = os.getenv("FLIPKART")
    client = pymongo.MongoClient(MONGODB_URI)
    mydb = client[FLIPKART]
    return mydb


def get_popular_items(db):
   
    # Load the dataset
    #db = initialising_mongoDB()
    top5 = db.Top5Products

    # retrieving the top5 products
    popular_items = []
    top_products = top5.find().sort("User rating for the product", -1)

    for index, product in enumerate(top_products, start=1):
        item_details = f"{index}. {product['product_name']} at {INR}{product['discounted_price']} \n\n Description: {product.get('description', 'No description available')} \n\n"
        popular_items.append(item_details)

    
    # Join all item details into a single string
    response_text = "Here are these week's popular items:\n" + "\n".join(popular_items)
    response_text += "\n\nWould you like to know more about any of these items? If not, please provide me the description of the item you are looking for."

    return response_text





""" KEYWORD DETECTION FUNCTION """

import nltk
from rake_nltk import Rake
from nltk.corpus import words, wordnet
nltk.download('words')
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')
# Function to check if the user's input is valid
def is_valid_input(user_input, valid_user_ids, keywords):
    # Convert both user IDs and keywords to set for fast membership checking
    keywords = set(word.lower() for word in keywords)

    # Tokenize and validate
    tokens = nltk.word_tokenize(user_input)
    valid_tokens = [word for word in tokens if word.lower() in wordnet.words() or word in valid_user_ids or word.lower() in keywords]

    return len(valid_tokens) > 0

def extract_keywords(item):
    r = Rake()
    r.extract_keywords_from_text(item)
    query_keyword = r.get_ranked_phrases_with_scores()
    query_keyword_ls = [keyword[1] for keyword in query_keyword]
    return query_keyword_ls






""" CHAT BOT FUNCTION"""

import re
from recSys.weighted import hybrid_recommendations

# Getting user intention
def getting_user_intention(user_input, intention_chain, previous_intention):
    user_intention = intention_chain.invoke({"input": user_input, "previous_intention": previous_intention})
    return user_intention
  
# Getting bot response
def getting_bot_response(user_intention, chain2, db, lsa_matrix, user_id):
    item_availability_match = re.search(r'Available in Store:\s*(.+)', user_intention)
    item_availability = item_availability_match.group(1)

    if item_availability != "Yes.":
        response = re.search(r'Suggested Actions or Follow-Up Questions:\s*(.+)', user_intention, re.DOTALL)
        bot_response = response.group(1).strip()
    else:
        match = re.search(r'Actionable Goal \+ Specific Details:\s*(.+)', user_intention)
        item = match.group(1)
        


        # calling hybrid_recommendations function 
        n_recommendations = 5  # number of recommendations to output (adjustable later)

        recommendations = hybrid_recommendations(
            catalogue = db.test_catalogue,    
            item = item, 
            user_id = user_id,  
            orderdata = db.users, 
            lsa_matrix = lsa_matrix,
            content_weight = 0.6, 
            collaborative_weight = 0.4,
            n_recommendations = n_recommendations, 
            
        )

        recommendations_text = "\n".join(
            f"**{idx + 1}. {rec['product_name']}** - Predicted Ratings: {rec['predicted_rating']:.2f}"
            for idx, rec in enumerate(recommendations)
        )

        # Getting follow-up questions from previous LLM
        questions_match = re.search(r'Suggested Actions or Follow-Up Questions:\s*(.+)', user_intention, re.DOTALL)
        questions = questions_match.group(1).strip()
        bot_response = chain2.invoke({"recommendations": recommendations_text, "questions": questions})


    return bot_response

