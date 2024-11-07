""" DATABASE FUNCTION"""

# Initializing data
import pandas as pd
from pymongo import MongoClient
import pymongo
import os
from datetime import datetime, timedelta
from forex_python.converter import CurrencyCodes
from dotenv import load_dotenv
from supabase import create_client

currency = CurrencyCodes()
INR = currency.get_symbol('INR')

def initialising_supabase():
    load_dotenv()
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_API_KEY = os.getenv("SUPABASE_API_KEY")
    supabase = create_client(SUPABASE_URL, SUPABASE_API_KEY)
    return supabase

def initialising_mongoDB():
    load_dotenv()
    MONGODB_URI = os.getenv("MONGODB_URI")
    FLIPKART = os.getenv("FLIPKART")
    client = pymongo.MongoClient(MONGODB_URI)
    mydb = client[FLIPKART]
    return mydb

def load_product_data():
    supabase = initialising_supabase()
    # Load data from the flipkart_cleaned table in supabase
    catalogue_data = pd.DataFrame(supabase.table('flipkart_cleaned').select('*').execute().data)

    # Create the 'content' column by concatenating 'description' and 'product_specifications'
    catalogue_data['content'] = catalogue_data['description'].astype(str) + ' ' + catalogue_data['product_specifications'].astype(str)
     # Ensure there are no NaN values which can cause issues
    catalogue_data['content'] = catalogue_data['content'].fillna('') 
    print("Successfully loaded DataFrame from Supabase")
 
    return catalogue_data
 



def get_popular_items(db):
   
    # Load the dataset
    supabase = initialising_supabase()
    top5 = pd.DataFrame(supabase.table('top5products').select('*').execute().data) 

    # retrieving the top5 products
    popular_items = []
    # top_products = top5.find().sort("User rating for the product", -1)

    for index, product in top5.iterrows():
        item_details = f"{index + 1}. {product['product_name']} at {INR}{product['discounted_price']} \n\n Description: {product.get('description', 'No description available')} \n\n"
        popular_items.append(item_details)

    
    # Join all item details into a single string
    response_text = "Here are these week's popular items:\n" + "\n".join(popular_items)
    response_text += "\n\nWould you like to know more about any of these items? If not, please provide me the description of the item you are looking for."

    return response_text





""" KEYWORD DETECTION FUNCTION """

import nltk
from rake_nltk import Rake
from nltk.corpus import words, wordnet
"""nltk.download('words')
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')"""
# Function to check if the user's input is valid
def is_valid_input(user_input, valid_user_ids, keywords):
    # Convert both user IDs and keywords to set for fast membership checking
    keywords = set(word.lower() for word in keywords)

    # Tokenize and validate
    tokens = nltk.word_tokenize(user_input)

    # Define validity check
    valid_tokens = [word for word in tokens if word.lower() in wordnet.words() or word in valid_user_ids or word.lower() in keywords]

    return len(valid_tokens) > 0

def extract_keywords(item):
    r = Rake()
    r.extract_keywords_from_text(item)
    query_keyword = r.get_ranked_phrases_with_scores()
    query_keyword_ls = [keyword[1] for keyword in query_keyword]
    return query_keyword_ls


# function to change string to dictionary with chatbot output
def parse_user_intention(user_intention_dictionary):
    dictionary = {}
    lines = user_intention_dictionary.split("\n")
    current_key = None

    for line in lines:
        # Remove leading dashes and strip any extra whitespace from the line
        cleaned_line = line.lstrip('- ').strip()
        if ": " in cleaned_line:  # Check if the line has a colon and space, indicating a key-value pair
            key, value = cleaned_line.split(": ", 1)
            current_key = key.strip()
            dictionary[current_key] = value.strip()
        elif current_key:  # This line might be a continuation of the last key's value
            dictionary[current_key] += " " + line.strip()
    
    return dictionary

def get_dummy_recommendation(keywords_list): # getting the top 3 products based on keywords
    return "hello"




""" CHAT BOT FUNCTION"""

import re
from recSys.weighted import hybrid_recommendations

# Getting user intention
def getting_user_intention_dictionary(user_input, intention_chain, previous_intention, past_follow_up_questions):
    if past_follow_up_questions is None:
        past_follow_up_questions = []

    
    user_intention_dictionary = intention_chain.invoke({"input": user_input, "previous_intention": previous_intention, "follow_up_questions": past_follow_up_questions})

    return user_intention_dictionary
  
# Getting bot response
def getting_bot_response(user_intention_dictionary, chain2, lsa_matrix, user_id):
    # Fetch the catalogue & users data from Supabase
    supabase = initialising_supabase()
    catalogue = load_product_data()
    users_data = supabase.table('synthetic_v2').select('*').execute().data

    item_availability = user_intention_dictionary.get("Available in Store")
    
    if item_availability == "No":
        print("Item not available")
        bot_response = user_intention_dictionary.get("Follow-Up Question")

    else: 
        fields_incomplete = int(user_intention_dictionary.get("Fields Incompleted"))

        if fields_incomplete > 2:
            print("Fields incomplete")
            bot_response = user_intention_dictionary.get("Follow-Up Question")

        else:
            print("Roughly complete")
            item = user_intention_dictionary.get("Product Item")



        # calling hybrid_recommendations function 
        #n_recommendations = 5  # number of recommendations to output (adjustable later)

            """recommendations = hybrid_recommendations(    
                item = item, 
                user_id = user_id,  
                orderdata = users_data,  
                lsa_matrix = lsa_matrix,
                content_weight = 0.6, 
                collaborative_weight = 0.4,
                n_recommendations = n_recommendations, 
                
            )"""
            print(item)
            recommendations =  get_dummy_recommendation(item)

            """recommendations_text = "\n".join(
                f"**{idx + 1}. {rec['product_name']}** - Predicted Ratings: {rec['predicted_rating']:.2f}"
                for idx, rec in enumerate(recommendations)
            )
    """
            # Getting follow-up questions from previous LLM
            questions = user_intention_dictionary.get("Follow-Up Question")
            bot_response = chain2.invoke({"recommendations": recommendations, "questions": questions})


    return bot_response



'''
def getting_bot_response(user_intention_dictionary, chain2, db, lsa_matrix, user_id):
    item_availability = user_intention_dictionary.get("Available in Store")
    

    if item_availability == "No":
        print("Item not available")
        bot_response = user_intention_dictionary.get("Follow-Up Question")

    else: 
        fields_incomplete = int(user_intention_dictionary.get("Fields Incompleted"))

        if fields_incomplete > 2:
            print("Fields incomplete")
            bot_response = user_intention_dictionary.get("Follow-Up Question")

        else:
            print("Roughly complete")
            item = user_intention_dictionary.get("Product Item")



        # calling hybrid_recommendations function 
        #n_recommendations = 5  # number of recommendations to output (adjustable later)

            """recommendations = hybrid_recommendations(
                catalogue = db.catalogue,    
                item = item, 
                user_id = user_id,  
                orderdata = db.users, 
                lsa_matrix = lsa_matrix,
                content_weight = 0.6, 
                collaborative_weight = 0.4,
                n_recommendations = n_recommendations, 
                
            )"""
            print(item)
            recommendations =  get_dummy_recommendation(item)

            """recommendations_text = "\n".join(
                f"**{idx + 1}. {rec['product_name']}** - Predicted Ratings: {rec['predicted_rating']:.2f}"
                for idx, rec in enumerate(recommendations)
            )
    """
            # Getting follow-up questions from previous LLM
            questions = user_intention_dictionary.get("Follow-Up Question")
            bot_response = chain2.invoke({"recommendations": recommendations, "questions": questions})


    return bot_response
'''
