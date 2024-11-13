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

def load_product_data(supabase):

    # Load data from the flipkart_cleaned table in supabase
    catalogue_data = supabase.table('flipkart_cleaned').select('*').execute().data
    """
    # Create the 'content' column by concatenating 'description' and 'product_specifications'
    catalogue_data['content'] = catalogue_data['description'].astype(str) + ' ' + catalogue_data['product_specifications'].astype(str)
     # Ensure there are no NaN values which can cause issues
    catalogue_data['content'] = catalogue_data['content'].fillna('') 
    print("Successfully loaded DataFrame from Supabase")"""
 
    return catalogue_data

def load_users_data(supabase): 
    users_data = pd.DataFrame(supabase.table('synthetic_v2').select('*').execute().data)
    return users_data

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


# Getting User profile and User purchase history
def getting_user_purchase_dictionary(user_id, supabase):
    user_profile = supabase.table('synthetic_v2').select('User Age', 'User Occupation', 'User Interests').eq('User ID', user_id).execute().data

    user_purchase_data = supabase.table('synthetic_v2').select('uniq_id', 'User rating for the product') \
                        .eq('User ID', user_id).execute().data
    
    
    # Check if purchase history exists
    if not user_purchase_data:
        return f"No purchase history found for user ID {user_id}."

    # Extract product IDs from purchase history
    product_ids = [purchase['uniq_id'] for purchase in user_purchase_data]

    # Query Supabase for product details based on the extracted product IDs
    product_details = supabase.table('flipkart_cleaned') \
                        .select('product_name, uniq_id') \
                        .in_('uniq_id', product_ids) \
                        .execute().data

    purchase_history_merge = pd.merge(pd.DataFrame(user_purchase_data), pd.DataFrame(product_details), how='inner', on='uniq_id')
    

    # Process each purchase and its product details
    user_purchases = []
    for items in purchase_history_merge.iterrows():
        user_purchases.append((items[1]['product_name'], items[1]['User rating for the product']))
        
    
    print("line 94, user_purchases: ", user_purchases)
    return user_profile, user_purchases
   
    


""" KEYWORD DETECTION FUNCTION """

import nltk
from rake_nltk import Rake
from nltk.corpus import words, wordnet

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
        "boteh2pghggcuphh": {
            "price": 1599,
            "product_name": "Rastogi Handicrafts JOINT LESS LEAK PROOF DECORATIVE 950 ml Bottle",
            "description": "AAA",
            "overall_rating": 5
        },
        "mngejhg7yhyzgugh": {
            "price": 1599,
            "product_name": "IPHONE 16 pro max",
            "description": "BBB",
            "overall_rating": 4
        }, 
        "tieee2ysk3faq5zf": {
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




""" CHAT BOT FUNCTION"""

import re
from recSys.weighted import hybrid_recommendations

# Getting user intention
def getting_user_intention_dictionary(user_input, intention_chain, previous_intention, past_follow_up_questions,  items_recommended):
    if past_follow_up_questions is None:
        past_follow_up_questions = []

    
    user_intention_dictionary = intention_chain.invoke({"input": user_input, "previous_intention": previous_intention, "follow_up_questions": past_follow_up_questions, "items_recommended": items_recommended})

    return user_intention_dictionary


"""TO BE REMOVED ONCE MERGED WITH RS"""

def get_item_details(uniq_ids ):
    # Initialize Supabase connection
    supabase = initialising_supabase()
    
    # Fetch full product details based on the uniq_ids from Supabase
    product_data = (
        supabase
        .table('flipkart_cleaned')
        .select('product_name, brand, retail_price, discounted_price, discount, description, product_specifications, overall_rating')
        .eq('pid', uniq_ids)
        .execute()
    )
    

    # Convert the results into a DataFrame
    product_df = pd.DataFrame(product_data.data)
        # Assuming details contain at least one item, and we are interested in the first one for demonstration
      # get the first product in the list
    readable_output = (
        f"Product Name: {product_df['product_name']}\n"
        f"Brand: {product_df['brand']}\n"
        f"Price: ₹{product_df['discounted_price']}\n"
        f"Rating: {product_df['overall_rating']}\n"
        f"Description: {product_df['description']}\n"
            
    )
    
    
    return readable_output

def getting_bot_response(user_intention_dictionary, chain2, supabase, user_profile, user_purchases, user_id):
  
    # Fetch the catalogue & users data from Supabase

    catalogue = (supabase.table('flipkart_cleaned'))
    users_data = (supabase.table('synthetic_v2').select('*').execute().data)

    item_availability = user_intention_dictionary.get("Available in Store")
    
    if item_availability == "No":
        print("Item not available")
        bot_response = user_intention_dictionary.get("Follow-Up Question")
        return None, bot_response
        
    # Related to Recommendation
    elif user_intention_dictionary.get("Related to Recommendation") == "Yes":
        
        product_id = user_intention_dictionary.get("Product ID")
        follow_up = user_intention_dictionary.get("Follow-Up Question")
        follow_up = user_intention_dictionary.get("Follow-Up Question")
        #print(product_id)
        item_recommendation = get_item_details(product_id)
        item_recommendation += "\n\n" + follow_up
        item_recommendation = get_item_details(product_id)
        item_recommendation += "\n\n" + follow_up
        return None, item_recommendation

    else: 
        # Set fields_incomplete to 0 if "Fields Incompleted" is None or not in the dictionary
        fields_incomplete = int(user_intention_dictionary.get("Fields Incompleted",0))
        item = user_intention_dictionary.get("Product Item")         
        keen_to_share = user_intention_dictionary.get("Keen to Share")
        

        # Check if all fields are incomplete and user prefers not to share more details
        if fields_incomplete == 3 and keen_to_share == "No":
            recommendations = get_dummy_recommendation(item)
            questions = user_intention_dictionary.get("Follow-Up Question")
<<<<<<< HEAD
            bot_response = chain2.invoke({"recommendations": recommendations, "questions": questions, "user_profile": user_profile, "user_purchase_history": user_purchases})
=======
            bot_response = chain2.invoke({"recommendations": recommendations, "questions": questions})
>>>>>>> 47078578458564437e3de5b80aff5ac1cd9efe79
           
        # Case where user has incomplete fields but is willing to share more preferences
        elif fields_incomplete == 3 and keen_to_share == "Yes":
            recommendations = None
            bot_response = user_intention_dictionary.get("Follow-Up Question")
            
        else:
            # Generate recommendations based on known preferences
            recommendations = hybrid_recommendations(user_intention_dictionary, user_id)
            recommendations = recommendations.to_dict(orient='records')
            # Getting follow-up questions from previous LLM if available
            questions = user_intention_dictionary.get("Follow-Up Question")
<<<<<<< HEAD
            bot_response = chain2.invoke({"recommendations": recommendations, "questions": questions,"user_profile": user_profile, "user_purchase_history": user_purchases})
=======
            bot_response = chain2.invoke({"recommendations": recommendations, "questions": questions})
>>>>>>> 47078578458564437e3de5b80aff5ac1cd9efe79
           
    return recommendations, bot_response
 

