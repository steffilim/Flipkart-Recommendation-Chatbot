import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from supabase import create_client
from weighted import hybrid_recommendations

def initialising_supabase():
    load_dotenv()
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_API_KEY = os.getenv("SUPABASE_API_KEY")
    supabase = create_client(SUPABASE_URL, SUPABASE_API_KEY)
    return supabase

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
    sampled_users = random.sample(list(user_ids), 5)
    return sampled_users

formatted_queries = [
    {"Product Item": "watch", "Budget": "No preference", "Brand": "No preference", "Product Details": "No preference"},
    {"Product Item": "skirt", "Budget": "No preference", "Brand": "No preference", "Product Details": "No preference"},
    {"Product Item": "battery", "Budget": "No preference", "Brand": "No preference", "Product Details": "No preference"},
    {"Product Item": "boots", "Budget": "No preference", "Brand": "No preference", "Product Details": "No preference"},
    {"Product Item": "necklace", "Budget": "No preference", "Brand": "No preference", "Product Details": "No preference"},
    {"Product Item": "watch", "Budget": "7000", "Brand": "No preference", "Product Details": "No preference"},
    {"Product Item": "skirt", "Budget": "No preference", "Brand": "No preference", "Product Details": "red"},
    {"Product Item": "battery", "Budget": "5000", "Brand": "No preference", "Product Details": "No preference"},
    {"Product Item": "boots", "Budget": "No preference", "Brand": "No preference", "Product Details": "black"},
    {"Product Item": "necklace", "Budget": "No preference", "Brand": "No preference", "Product Details": "silver"}
]

sampled_users = sampling_of_users(order_data)

results = []

# Loop over each user and each query
for user_id in sampled_users:
    for query in formatted_queries:
        # Call hybrid_recommendations for the current user and query
        print("this is the query:")
        print(query)
        recommendations = hybrid_recommendations(
            extracted_info=query,
            user_id=user_id,
            content_weight=20,
            collaborative_weight=0.5,
            brand_preference=query.get("Brand"),
            specs_preference=query.get("Product Details"),
            top_n=10
        )
        print(recommendations)
        # Add each recommendation to the results
        for rank, (_, row) in enumerate(recommendations.iterrows(), start=1):
            results.append({
                "User ID": user_id,
                "Query": query,
                "Product ID": row['uniq_id'],
                "Recommended Product": row['product_name'],
                "Product Description": row['description'],
                "Product Specification": row['product_specifications'],
                "Rank": rank
            })

def load_rankings_data():
    supabase = initialising_supabase()
    rankings_data = pd.DataFrame(supabase.table('rankings').select('*').execute().data)
    print("Successfully loaded ranking data from Supabase")
    return rankings_data

ranking_data = load_rankings_data()

def calculate_dcg_binary(relevance, rank):
    if np.sum(relevance) == 0:
        return 0.0
    return np.sum(relevance / np.log2(rank + 1))


def calculate_ndcg_binary(ranking_data):
    # Ensure Relevance and Rank columns are numeric
    ranking_data["Relevance"] = pd.to_numeric(ranking_data["Relevance"], errors='coerce').fillna(0)
    ranking_data["Rank"] = pd.to_numeric(ranking_data["Rank"], errors='coerce').fillna(0)

    scores = {}
    for user, group in ranking_data.groupby("User ID"):
        relevance = group["Relevance"].values
        rank = group["Rank"].values
        dcg = calculate_dcg_binary(relevance, rank)
        idcg = calculate_dcg_binary(np.sort(relevance)[::-1], np.arange(1, len(relevance) + 1))

        ndcg = dcg / idcg if idcg > 0 else 0.0

        scores[user] = {
            "DCG": dcg,
            "IDCG": idcg,
            "NDCG": ndcg,
        }

    return scores

scores = calculate_ndcg_binary(ranking_data)
print(scores)