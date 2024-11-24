from surprise import Dataset, Reader
from surprise import SVD, NMF
from surprise.model_selection import cross_validate, train_test_split, GridSearchCV
import pandas as pd
import numpy as np
import sys
import os
from functions.databaseFunctions import *

def filter_products(product_name=None, price_limit=None, brand=None, product_specifications=None):
    """
    Filters products from the database based on given criteria.

    Args:
        product_name (str, optional): The name or part of the name of the product to search for.
        price_limit (float, optional): The maximum price of the product.
        brand (str, optional): The brand name to filter by.
        product_specifications (str, optional): Specific product details to match.

    Returns:
        list[dict]: A list of dictionaries containing product details that match the criteria.
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
    Fetches products from the database based on extracted user preferences.

    Args:
        extracted_info (dict): A dictionary containing user preferences with keys:
            - "Product Item" (str): The product name or category.
            - "Budget" (float or str): The maximum budget for the product.
            - "Brand" (str): The preferred brand.
            - "Product Details" (str): Specific product features or specifications.

    Returns:
        pd.DataFrame: A DataFrame containing the filtered products.
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
    return pd.DataFrame(filtered_products)

def svd_recommend_surprise(user_id, catalogue, user_intention_dictionary, n_recommendations=20):
    """
    Recommends products to a user based on collaborative filtering using the SVD algorithm.

    Args:
        user_id (str): The ID of the user for whom recommendations are generated.
        catalogue (pd.DataFrame): A DataFrame containing product details (catalogue).
        user_intention_dictionary (dict): A dictionary of user preferences with keys:
            - "Product Item" (str): The product name or category.
            - "Budget" (float or str): The maximum budget for the product.
            - "Brand" (str): The preferred brand.
            - "Product Details" (str): Specific product features or specifications.
        n_recommendations (int, optional): The number of recommendations to generate. Defaults to 20.

    Returns:
        pd.DataFrame: A DataFrame containing the recommended product IDs and their predicted ratings.
    """
    
    print("SVD")
    supabase = initialising_supabase()
     # Fetch the catalogue & users data from Supabase
    catalogue = pd.DataFrame(catalogue)
    orderdata = load_order_data()

    # Initialize an empty DataFrame with the correct schema
    recommendations_df = pd.DataFrame(columns=['uniq_id', 'predicted_rating'])

    reader = Reader(rating_scale=(0, 5))

    # Get dataset
    dataset = Dataset.load_from_df(orderdata[['User ID', 'uniq_id', 'User rating for the product']], reader)
    trainset = dataset.build_full_trainset()

    #parameters from grid search
    params = {'n_factors': 80, 'n_epochs': 5, 'lr_all': 0.002, 'reg_all': 0.6}
    # Build SVD
    svd = SVD(n_factors=params['n_factors'], 
            n_epochs=params['n_epochs'], 
            lr_all=params['lr_all'], 
            reg_all=params['reg_all'])
    svd.fit(trainset)

    filtered_products_df = fetch_filtered_products(user_intention_dictionary)

    user_ratings = orderdata[orderdata['User ID'] == user_id]
    user_rated_products = user_ratings['uniq_id'].values
    unrated_products = filtered_products_df[~filtered_products_df['uniq_id'].isin(user_rated_products)]['uniq_id'].unique()

    predictions = []
    for product_id in unrated_products:
        pred = svd.predict(user_id, product_id)
        predictions.append((product_id, pred.est))

    # Sort predictions by estimated rating
    sorted_predictions = sorted(predictions, key=lambda x: x[1], reverse=True)

    # Select top N recommendations based on sorted predictions
    recommended_product_ids = [prod[0] for prod in sorted_predictions[:n_recommendations]]

    # Filter catalogue based on recommended product IDs and add predicted ratings
    recommendations = catalogue[catalogue['uniq_id'].isin(recommended_product_ids)].copy()

    recommendations['predicted_rating'] = recommendations['uniq_id'].map(dict(predictions))

    # Return the final DataFrame with recommendations
    recommendations_df = recommendations[['uniq_id', 'predicted_rating']]

    return recommendations_df

# user_intention_dictionary = {
#     "Product Item": "skirt",
#     "Budget": "2000",
#     "Brand": "",
#     "Product Details": "No preference"
# }
# user_id = 'U01394'
# recommendations = svd_recommend_surprise(user_id, catalogue, user_intention_dictionary, n_recommendations=20)
# print(recommendations)