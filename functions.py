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

import random
def get_dummy_recommendation(keywords_list): # getting the top 3 products based on keywords
    products = {
        "JSBFKS234KB": {
            "price": 1599,
            "product_name": "Rastogi Handicrafts JOINT LESS LEAK PROOF DECORATIVE 950 ml Bottle",
            "description": "AAA",
            "overall_rating": 5
        },
        "NGEORI1otOÅ": {
            "price": 1599,
            "product_name": "IPHONE 16 pro max",
            "description": "BBB",
            "overall_rating": 4
        }, 
        "JKJKJKJKJWKERLW": {
            "price": 1599,
            "product_name": "SPORTS running shoes",
            "description": "BBB",
            "overall_rating": 4
        }

    }

    selected_keys = random.sample(list(products.keys()), 2)
    
    # Construct a new dictionary with only the selected items
    selected_products = {key: products[key] for key in selected_keys}
    
    return selected_products


    return products



""" CHAT BOT FUNCTION"""

import re
from recSys.weighted import hybrid_recommendations

# Getting user intention
def getting_user_intention_dictionary(user_input, intention_chain, previous_intention, past_follow_up_questions, bot_response, items_recommended):
    if past_follow_up_questions is None:
        past_follow_up_questions = []

    
    user_intention_dictionary = intention_chain.invoke({"input": user_input, "previous_intention": previous_intention, "follow_up_questions": past_follow_up_questions, "bot_response": "hello", "items_recommended": items_recommended})

    return user_intention_dictionary


def get_item_details(db, pid, follow_up_question):

    """
    SUPABASE IMPLEMENTATION TO GO HERE
    BUT THE FORMAT IS AS FOLLOWS:
    product name:
    brand: 
    description:
    overall rating of the product:
    price:
    """

    return follow_up_question
  
# Getting bot response
def getting_bot_response(user_intention_dictionary, chain2, db, user_input, user_id):
    item_availability = user_intention_dictionary.get("Available in Store")
    
    
    # for unavailable items
    if item_availability == "Not Available":
        print("Item not available")
        bot_response = user_intention_dictionary.get("Follow-Up Question")
        return None, bot_response

    # for available items
    
    # Related to Recommendation
    elif user_intention_dictionary.get("Related to Recommendation") == "Yes":
        
        product_id = user_intention_dictionary.get("Product ID")
        #print(product_id)
        item_recommendation = get_item_details(db, product_id, user_intention_dictionary.get("Follow-Up Question"))

        return None, item_recommendation
    
    else: 
        fields_incomplete = sum(1 for key, value in user_intention_dictionary.items() if value == "Not specified")
        #fields_incomplete = (user_intention_dictionary.get("Fields Incompleted"))

        if fields_incomplete > 2:
            print("Fields incomplete")
            bot_response = user_intention_dictionary.get("Follow-Up Question")
            return None, bot_response

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
            
            recommendations =  get_dummy_recommendation(item)
            print(recommendations)

            """recommendations_text = "\n".join(
                f"**{idx + 1}. {rec['product_name']}** - Predicted Ratings: {rec['predicted_rating']:.2f}"
                for idx, rec in enumerate(recommendations)
            )
    """
            # Getting follow-up questions from previous LLM
            questions = user_intention_dictionary.get("Follow-Up Question")
            bot_response = chain2.invoke({"recommendations": recommendations, "questions": questions})
            print(recommendations)

            return recommendations, bot_response

