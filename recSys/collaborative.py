from surprise import Dataset, Reader
from surprise import SVD, NMF
from surprise.model_selection import cross_validate, train_test_split, GridSearchCV
import pandas as pd
import numpy as np
import sys
import os
from functions.databaseFunctions import *

def filter_products(product_name=None, price_limit=None, brand=None, product_specifications=None):
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
    # ensure correct inputs
    # ensure correct type
    # if cannot find in dictionary, just none
    # Extract values from the dictionary with default values for any missing keys
    product_name = extracted_info.get("Product Item")
    price_limit = extracted_info.get("Budget")
    brand = extracted_info.get("Brand")
    product_specifications = extracted_info.get("Product Details")

    # print(product_name, price_limit, brand, product_specifications)

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
    print("SVD")
    supabase = initialising_supabase()
     # Fetch the catalogue & users data from Supabase
    catalogue = pd.DataFrame(catalogue)
    orderdata = load_order_data()

    # Initialize an empty DataFrame with the correct schema
    recommendations_df = pd.DataFrame(columns=['uniq_id', 'predicted_rating'])


    reader = Reader(rating_scale=(0, 5))

    #print(orderdata)
    #print(reader)
    dataset = Dataset.load_from_df(orderdata[['User ID', 'uniq_id', 'User rating for the product']], reader)
    trainset = dataset.build_full_trainset()

    svd = SVD()
    svd.fit(trainset)

    filtered_products_df = fetch_filtered_products(user_intention_dictionary)

    user_ratings = orderdata[orderdata['User ID'] == user_id]
    user_rated_products = user_ratings['uniq_id'].values
    unrated_products = filtered_products_df[~filtered_products_df['uniq_id'].isin(user_rated_products)]['uniq_id'].unique()

    predictions = []
    for product_id in unrated_products:
        pred = svd.predict(user_id, product_id)
        predictions.append((product_id, pred.est))
    print("collab.py predictions ", predictions)

    # Sort predictions by estimated rating
    sorted_predictions = sorted(predictions, key=lambda x: x[1], reverse=True)
    print("collab.py sorted predictions ", sorted_predictions)

    # Select top N recommendations based on sorted predictions
    recommended_product_ids = [prod[0] for prod in sorted_predictions[:n_recommendations]]
    print("line 108", type(recommended_product_ids))

    # Filter catalogue based on recommended product IDs and add predicted ratings
    recommendations = catalogue[catalogue['uniq_id'].isin(recommended_product_ids)].copy()
    print("line 112")
    recommendations['predicted_rating'] = recommendations['uniq_id'].map(dict(predictions))
    print("line 114")

    # Return the final DataFrame with recommendations
    recommendations_df = recommendations[['uniq_id', 'predicted_rating']]
    print("collab.py line 115: ", recommendations_df)
    print("collab.py line 116: ", type(recommendations_df))

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