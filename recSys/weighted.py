from recSys.collaborative import svd_recommend_surprise
from recSys.contentBased import recommend_top_products 

import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
from dotenv import load_dotenv
import os
from concurrent.futures import ThreadPoolExecutor
from sklearn.preprocessing import MinMaxScaler 
import time
import random
from functions.databaseFunctions import *

''' filter products '''
def filter_products(product_name=None, price_limit=None, brand=None, product_specifications=None):
    """
    Filters products from the database based on the specified criteria.

    Args:
        product_name (str, optional): The name or part of the product name to search for.
        price_limit (float, optional): The maximum price for the products.
        brand (str, optional): The brand of the products to search for.
        product_specifications (str, optional): Additional specifications to filter products.

    Returns:
        list[dict]: A list of dictionaries containing the filtered product data.
    
    Raises:
        Exception: If there's an error in the database query or connection.
    """

    # Build the SQL query dynamically based on the filters provided
    supabase = initialising_supabase()
    query = supabase.table("flipkart_cleaned_2k").select("*")
    
    if product_name and isinstance(product_name, str) and product_name.strip():
        query = query.ilike("product_name", f"%{product_name}%")
    
    if price_limit is not None:
        query = query.lte("retail_price", price_limit)
    
    if brand and isinstance(brand, str) and brand.strip():
        query = query.ilike("brand", f"%{brand}%")
    
    if product_specifications and isinstance(product_specifications, str) and product_specifications.strip():
        query = query.ilike("product_specifications", f"%{product_specifications}%")
    
    # Execute the query and get the results
    response = query.execute()
    return response.data

def fetch_filtered_products(extracted_info):
    """
    Fetches filtered products based on extracted information.

    Args:
        extracted_info (dict): Dictionary containing user preferences such as 
                               'Product Item', 'Budget', 'Brand', and 'Product Details'.

    Returns:
        list[dict]: A list of dictionaries containing the filtered product data.
    """

    # Extract values from the dictionary with default values for any missing keys
    product_name = extracted_info.get("Product Item")
    price_limit = extracted_info.get("Budget")
    brand = extracted_info.get("Brand")
    product_specifications = extracted_info.get("Product Details")

    # Convert "Not specified" to None for each field
    if product_name == "No preference":
        product_name = None
    if price_limit == "No preference":
        price_limit = None
    if brand == "No preference":
        brand = None
    if product_specifications == "No preference":
        product_specifications = None

    if price_limit is not None and price_limit != "Not specified":
        try:
            price_limit = float(price_limit)
        except ValueError:
            print("Warning: price_limit is not a valid number:", price_limit)
            price_limit = None

    filtered_products = filter_products(
        product_name=product_name,
        price_limit=price_limit,
        brand=brand,
        product_specifications=product_specifications
    )
    print("filtered length", len(filtered_products))
    return filtered_products

''' helper functions '''
def get_user_query(extracted_info, keys=['product_name', 'brand', 'specifications'], separator=', '):
    """
    Constructs a user query string based on provided keys and values in extracted_info.

    Args:
        extracted_info (dict): Dictionary of extracted user preferences.
        keys (list[str], optional): List of keys to include in the query. Defaults to ['product_name', 'brand', 'specifications'].
        separator (str, optional): Separator to use between concatenated query terms. Defaults to ', '.

    Returns:
        str: A concatenated string of the user query.
    """

    result = ''
    
    # Iterate over the list of keys to extract values
    for key in keys:
        # Check if key exists and its value is not 'No preference'
        if key in extracted_info and extracted_info[key] != 'No preference':
            result += str(extracted_info[key]) + separator
    
    # Remove the trailing separator
    result = result.rstrip(separator)
    return result

# fetching only specified columns: 'uniq_id, product_name, brand, retail_price, discounted_price, description'
def get_product_details_from_supabase(uniq_ids, columns = None):
    """
    Fetch full product details from Supabase based on provided unique IDs and selected columns.

    Args:
        uniq_ids (list of str): List of unique product IDs to fetch details for.
                                If empty, an empty DataFrame is returned.
        columns (list of str, optional): List of columns to select from the database.
                                         Defaults to common product detail columns.

    Returns:
        pandas.DataFrame: DataFrame containing the requested product details.
                          Returns an empty DataFrame if `uniq_ids` is empty.
    
    Raises:
        TypeError: If `uniq_ids` is not a list.
        ValueError: If any ID in `uniq_ids` is not a string or if `columns` is not a list.
    """

    # Ensure uniq_ids is a list
    if not isinstance(uniq_ids, list):
        raise TypeError("uniq_ids must be a list of strings.")

    # Return an empty DataFrame if uniq_ids is empty
    if not uniq_ids:
        return pd.DataFrame(columns=columns if columns else [])

    # Default columns if none are provided
    if columns is None:
        columns = ['uniq_id', 'product_name', 'brand', 'retail_price', 'discounted_price', 
                   'discount', 'description', 'product_specifications', 'overall_rating']
    
    # Ensure columns is a list of strings
    if not all(isinstance(col, str) for col in columns):
        raise ValueError("columns must be a list of strings.")

    # Initialize Supabase connection
    supabase = initialising_supabase()

    # Fetch full product details based on the uniq_ids from Supabase
    product_data = (
        supabase
        .table('flipkart_cleaned')
        .select(', '.join(columns))
        .in_('uniq_id', uniq_ids)
        .execute()
    )

    # Convert the results into a DataFrame
    product_df = pd.DataFrame(product_data.data)
    
    return product_df

''' content based '''
def fetch_content_recommendation(extracted_info):
    """
    Fetches content-based recommendations based on user query and filtered products.

    Args:
        extracted_info (dict): Dictionary of user preferences for filtering and recommendations.

    Returns:
        pandas.DataFrame: DataFrame of content-based recommendations.
    """

    print("Inside content recommender")

    supabase = initialising_supabase() 

    user_query = get_user_query(extracted_info)

    filtered_products = fetch_filtered_products(extracted_info)

    if len(filtered_products) > 40:
        filtered_products = random.sample(filtered_products, 40)

    content_recommendations = recommend_top_products(user_query, filtered_products)

    return content_recommendations

''' collaborative '''
def fetch_collaborative_recommendation(user_id, extracted_info):
    """
    Fetches collaborative filtering recommendations based on user data.

    Args:
        user_id (str): User ID for fetching collaborative filtering data.
        extracted_info (dict): User preferences for additional context.

    Returns:
        pandas.DataFrame: DataFrame of collaborative filtering recommendations.
    """

    print("Inside collaborative recommender")

    catalogue = load_product_data()

    collaborative_recommendations = svd_recommend_surprise(user_id, catalogue, extracted_info)

    return collaborative_recommendations

''' hybridisation helper functions '''
def normalize_collaborative_scores(collaborative_recommendations):
    """
    Normalizes collaborative recommendation scores using Min-Max scaling.

    Args:
        collaborative_recommendations (pandas.DataFrame): DataFrame of collaborative recommendations with 'predicted_rating'.

    Returns:
        pandas.DataFrame: Updated DataFrame with normalized 'predicted_rating'.
    """

    if not collaborative_recommendations.empty and 'predicted_rating' in collaborative_recommendations.columns:
        scaler = MinMaxScaler()
        collaborative_recommendations['normalized_predicted_rating'] = scaler.fit_transform(
            collaborative_recommendations[['predicted_rating']]
        )
    else:
        collaborative_recommendations['normalized_predicted_rating'] = 0  # Fallback
    return collaborative_recommendations

def calculate_final_scores(content_recommendations, collaborative_recommendations, content_weight=20, collaborative_weight=0.3, top_n=10):
    """
    Calculates final hybrid scores by combining content and collaborative recommendations.

    Args:
        content_recommendations (pandas.DataFrame): Content-based recommendation scores.
        collaborative_recommendations (pandas.DataFrame): Collaborative recommendation scores.
        content_weight (float, optional): Weight for content recommendation scores. Defaults to 20.
        collaborative_weight (float, optional): Weight for collaborative recommendation scores. Defaults to 0.3.
        top_n (int, optional): Number of top recommendations to return. Defaults to 10.

    Returns:
        pandas.DataFrame: DataFrame of top recommendations ranked by final score.
    """

    # Weight scores
    content_recommendations['weighted_similarity_score'] = (
        content_recommendations.get('similarity_score', 0) + 1
    ) * content_weight

    if 'normalized_predicted_rating' in collaborative_recommendations.columns:
        collaborative_recommendations['weighted_predicted_rating'] = (
            collaborative_recommendations['normalized_predicted_rating'] * collaborative_weight
        )
    else:
        collaborative_recommendations['weighted_predicted_rating'] = 0

    # Perform outer merge
    hybrid = pd.merge(content_recommendations, collaborative_recommendations, on='uniq_id', how='outer').fillna(0)

    hybrid['final_score'] = hybrid['weighted_similarity_score'] + hybrid['weighted_predicted_rating']
    return hybrid.nlargest(top_n, 'final_score')

''' hybrid recommendation system '''
def hybrid_recommendations(extracted_info, user_id, content_weight=20, collaborative_weight=0.5, top_n=10):
    """
    Generates hybrid recommendations by combining content-based and collaborative filtering approaches.

    Args:
        extracted_info (dict): User preferences for filtering and recommendations.
        user_id (str): User ID for collaborative filtering.
        content_weight (float, optional): Weight for content-based scores. Defaults to 20.
        collaborative_weight (float, optional): Weight for collaborative scores. Defaults to 0.5.
        top_n (int, optional): Number of top recommendations to return. Defaults to 10.

    Returns:
        pandas.DataFrame: DataFrame containing detailed information about the top recommended products.
    """

    # Step 1: Fetch filtered products and order data
    filtered_products = fetch_filtered_products(extracted_info)

    # check if filtered_products is empty
    if not filtered_products:
        print("no filtered products, returning empty dataframe")
        return pd.DataFrame()
    
    orderdata = load_order_data()

    # Step 2: Fetch content and collaborative recommendations concurrently
    with ThreadPoolExecutor() as executor:
        content_future = executor.submit(
            fetch_content_recommendation, extracted_info
        )
        collaborative_future = executor.submit(
            fetch_collaborative_recommendation, user_id, extracted_info
        )

        try:
            content_recommendations = content_future.result()
        except Exception as e:
            print(f"Error in content recommendations: {e}")
            content_recommendations = []  # Default or empty value if there's an error

        try:
            collaborative_recommendations = collaborative_future.result()
        except Exception as e:
            print(f"Error in collaborative recommendations: {e}")
            collaborative_recommendations = []  # Default or empty value if there's an error

    # Step 3: Normalize collaborative scores if available
    collaborative_recommendations = normalize_collaborative_scores(collaborative_recommendations)

    # Step 4: Calculate final scores and get top recommendations
    top_n_recommendations = calculate_final_scores(
        content_recommendations, collaborative_recommendations, content_weight, collaborative_weight, top_n
    )

    # Step 5: Fetch product details based on top recommendations
    uniq_ids = top_n_recommendations['uniq_id'].tolist()  # Extract 'uniq_id' as a list from DataFrame

    product_details_df = get_product_details_from_supabase(uniq_ids)

    print("hybrid rec sys items", product_details_df['product_name'] if not product_details_df.empty else "No products found")

    return product_details_df

'''test'''
# import time
# extracted_info = {
#     "Product Item": "skirt",
#     "Budget": "2000",
#     "Brand": "",
#     "Product Details": "No preference"
# }

# user_id = "U12345"
# content_weight = 0.9
# collaborative_weight = 0

# Start timer
# start_time = time.time()

# recs = hybrid_recommendations(extracted_info, user_id)
# recs

# # End timer
# end_time = time.time()
# # Calculate and print the elapsed time
# elapsed_time = end_time - start_time
# print(f"Execution time: {elapsed_time:.2f} seconds")