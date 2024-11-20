import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from supabase import create_client
from functions import initialising_supabase
from weighted import hybrid_recommendations

def load_order_data():
    """
    Loads order data from the Supabase database.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the order data.
    """

    supabase = initialising_supabase()
    order_data = pd.DataFrame(supabase.table('synthetic_v2_2k').select('*').execute().data)
    return order_data

order_data = load_order_data()

def load_product_data():
    """
    Loads product catalog data from the Supabase database.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the product data, 
                      with an additional 'content' column combining 'description' 
                      and 'product_specifications'.
    """

    supabase = initialising_supabase()

    # Load data from the flipkart_cleaned table in supabase
    catalogue_data = pd.DataFrame(supabase.table('flipkart_cleaned_2k').select('*').execute().data)

    # Create the 'content' column by concatenating 'description'
    catalogue_data['content'] = catalogue_data['description'].astype(str) + ' ' + catalogue_data['product_specifications'].astype(str)

    # Ensure there are no NaN values which can cause issues
    catalogue_data['content'] = catalogue_data['content']. fillna("")
    print("Successfully loaded product data from Supabase")
    return catalogue_data

product_data = load_product_data()

def sampling_of_users(order_data):
    """
    Samples a subset of user IDs from the order data.

    Args:
        order_data (pd.DataFrame): DataFrame containing the order data.

    Returns:
        list: A list of 20 randomly sampled unique user IDs.
    """

    random.seed(1234)
    user_ids = order_data['User ID'].unique()
    sampled_users = random.sample(list(user_ids), 20)
    return sampled_users

queries=[["laptop", "Acer"], ["skirt", "red"], ["phone", "Samsung"], ["furniture", "wood"], ["accessories", "silver"]]

def ranking(product, user_profile):
    """
    Calculates a ranking score for a product based on user preferences.

    Args:
        product (dict): A dictionary containing product attributes 
                        such as 'discounted_price', 'overall_rating', 
                        and 'product_category_tree'.
        user_profile (dict): A dictionary containing user preferences, 
                             such as 'User Interests'.

    Returns:
        tuple: A tuple of ranking criteria (negative price, rating, interest match).
    """
        
    discounted_price = -product['discounted_price']  # Negative to prioritize higher prices
    rating = product['overall_rating']
    matches_interest = user_profile['User Interests'] in product['product_category_tree']
    return (discounted_price, rating, matches_interest)

def generate_user_ranking(product_list, user_profile):
    """
    Generates a user-specific ranking of products based on their preferences.

    Args:
        product_list (list): List of dictionaries, each representing a product.
        user_profile (dict): Dictionary containing user-specific preferences.

    Returns:
        list: A list of product IDs ranked according to the user's preferences.
    """

    # Sort using the ranking_key function without lambda
    ranked_products = sorted(product_list, key=lambda product: ranking(product, user_profile))
    return [prod['uniq_id'] for prod in ranked_products]

ground_truth_data = []

def generate_ground_truth_data(order_data, sampled_users, queries, num_products=10):
    """
    Generates ground truth data for user-product ranking based on hybrid recommendations.

    Args:
        order_data (pd.DataFrame): DataFrame containing order data.
        sampled_users (list): List of sampled user IDs.
        queries (list): List of query keyword pairs (e.g., [["laptop", "Acer"]]).
        num_products (int, optional): Number of products to include in recommendations. Default is 10.

    Returns:
        pd.DataFrame: DataFrame containing ground truth data with user IDs, queries, 
                      generated ranked lists, and user-preferred rankings.

    Raises:
        ValueError: If inputs are invalid or if queries are not properly formatted.
    """

    ground_truth_data = []

    for user_id in sampled_users:
        # Get user profile from order data
        user_profile = order_data[order_data['User ID'] == user_id].iloc[0]
        
        for query_keywords in queries:
            query = " ".join(query_keywords)  # Convert list of keywords into a single query string
            
            initial_recommendations = hybrid_recommendations(
                item=query,
                user_id=user_id,
                orderdata=order_data,
                lsa_matrix=None,  # Pass your LSA matrix if applicable
                content_weight=0.5,
                collaborative_weight=0.5,
                brand_preference=user_profile.get('User Interests'),  # Example use of user profile
                specs_preference=None,
                n_recommendations=9
            )
            
            # Generate user-preferred ranking
            user_ranked_list = generate_user_ranking(initial_recommendations, user_profile)
            
            # Store ground truth data entry
            ground_truth_data.append({
                "user_id": user_id,
                "query": query,
                "generated_ranked_list": [prod['uniq_id'] for prod in initial_recommendations],
                "user_preferred_rank": user_ranked_list
            })

    # Convert to DataFrame and return
    ground_truth_df = pd.DataFrame(ground_truth_data)
    return ground_truth_df

sampled_users = sampling_of_users(order_data)
round_truth_df = generate_ground_truth_data(order_data, sampled_users, queries)