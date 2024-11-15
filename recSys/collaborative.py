from surprise import Dataset, Reader
from surprise import SVD, NMF
from surprise.model_selection import cross_validate, train_test_split, GridSearchCV
import pandas as pd
import numpy as np
from supabase import create_client
from dotenv import load_dotenv
import sys
import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from functions import initialising_supabase, load_product_data

"""
flipkart = pd.read_csv("newData/flipkart_cleaned.csv")
flipkart
flipkart['product_category_tree'] = flipkart['product_category_tree'].apply(lambda x: ' '.join(eval(x)).lower())

orderdata = pd.read_csv("newData/synthetic_v2.csv")
orderdata = orderdata.rename(columns={'Product ID': 'uniq_id'})
orderdata
"""

def initialising_supabase():
    load_dotenv()
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_API_KEY = os.getenv("SUPABASE_API_KEY")
    supabase = create_client(SUPABASE_URL, SUPABASE_API_KEY)
    return supabase

def load_product_data():
    supabase = initialising_supabase()
    # Load data from the flipkart_cleaned table in supabase
    catalogue_data = pd.DataFrame(supabase.table('flipkart_cleaned_2k').select('*').execute().data)

    # Create the 'content' column by concatenating 'description' and 'product_specifications'
    catalogue_data['content'] = catalogue_data['description'].astype(str) + ' ' + catalogue_data['product_specifications'].astype(str)
     # Ensure there are no NaN values which can cause issues
    catalogue_data['content'] = catalogue_data['content'].fillna('') 

    # print("Successfully loaded product data from Supabase")
 
    return catalogue_data
catalogue = load_product_data()
def load_order_data():
    supabase = initialising_supabase()
    # Load data from the flipkart_cleaned table in supabase
    order_data = pd.DataFrame(supabase.table('synthetic_v2_2k').select('*').execute().data)

    # print("Successfully loaded order from Supabase")
 
    return order_data
orderdata = load_order_data()
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

    # Print each field and its type after processing
    '''
    print("product_name:", product_name, "| Type:", type(product_name))
    print("price_limit:", price_limit, "| Type:", type(price_limit))
    print("brand:", brand, "| Type:", type(brand))
    print("overall_rating:", overall_rating, "| Type:", type(overall_rating))
    print("product_specifications:", product_specifications, "| Type:", type(product_specifications))
    '''

    if price_limit is not None and price_limit != "Not specified":
        try:
            price_limit = float(price_limit)
        except ValueError:
            print("Warning: price_limit is not a valid number:", price_limit)
            price_limit = None

    # print(product_name, price_limit, brand, product_specifications)

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
    catalogue = catalogue
    orderdata = load_order_data()

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
    print("these are the predictions:")
    print(predictions)

    # Sort predictions by estimated rating
    sorted_predictions = sorted(predictions, key=lambda x: x[1], reverse=True)
    print("Sorted predictions (top rated):")
    print(sorted_predictions)

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

def svd_recommend_surprise_filtered(user_id, extracted_info, n_recommendations=10):
    print("SVD - Filtered Recommendations")

    # Fetch the catalogue & users data from Supabase
    supabase = initialising_supabase()
    catalogue = load_product_data()

    # Create the dataset
    orderdata = load_order_data()

    reader = Reader(rating_scale=(0, 5))
    dataset = Dataset.load_from_df(orderdata[['User ID', 'uniq_id', 'User rating for the product']], reader)
    trainset = dataset.build_full_trainset()

    svd = SVD()
    svd.fit(trainset)

    # Filter unrated products
    user_ratings = orderdata[orderdata['User ID'] == user_id]
    user_rated_products = user_ratings['uniq_id'].values
    all_products = orderdata['uniq_id'].unique()
    unrated_products = [prod for prod in all_products if prod not in user_rated_products]

    predictions = []
    for product_id in unrated_products:
        pred = svd.predict(user_id, product_id)
        predictions.append((product_id, pred.est))

    sorted_predictions = sorted(predictions, key=lambda x: x[1], reverse=True)

    recommended_product_ids = [prod[0] for prod in sorted_predictions[:n_recommendations]]
    recommendations = catalogue[catalogue['uniq_id'].isin(recommended_product_ids)].copy()
    recommendations['predicted_rating'] = recommendations['uniq_id'].apply(
        lambda uid: next((pred[1] for pred in predictions if pred[0] == uid), None)
    )

    # # Get product information for the recommended product IDs
    # recommendations = catalogue[catalogue['uniq_id'].isin(recommended_product_ids)].copy()

    # # Add predicted ratings to the recommendations
    # for idx, row in recommendations.iterrows():
    #     prediction = next((pred[1] for pred in predictions if pred[0] == row['uniq_id']), None)
    #     recommendations.loc[idx, 'predicted_rating'] = prediction

    # Now apply filters based on extracted_info
    if extracted_info.get("product_name"):
        recommendations = recommendations[recommendations['product_name'].str.contains(extracted_info["product_name"], case=False)]
    
    if extracted_info.get("price_limit"):
        recommendations = recommendations[recommendations['retail_price'] <= extracted_info["price_limit"]]
    
    if extracted_info.get("brand"):
        recommendations = recommendations[recommendations['brand'].str.contains(extracted_info["brand"], case=False)]
    
    if extracted_info.get("overall_rating") and extracted_info["overall_rating"] != '0':
        recommendations = recommendations[recommendations['overall_rating'] >= float(extracted_info["overall_rating"])]
    
    if extracted_info.get("product_specifications"):
        recommendations = recommendations[recommendations['product_specifications'].str.contains(extracted_info["product_specifications"], case=False)]

    # Return the filtered recommendations
    recommendations_df = recommendations[['uniq_id', 'product_name', 'description', 'product_category_tree', 'predicted_rating', 'retail_price']]

    return recommendations_df





'''
def svd_recommend_surprise(catalogue, user_id, orderdata, n_recommendations=20):
    print("SVD")
    cursor_cat = catalogue.find({})
    catalogue = pd.DataFrame(list(cursor_cat))  

    cursor_order = orderdata.find({})   
    orderdata = pd.DataFrame(list(cursor_order))
    reader = Reader(rating_scale=(0, 5))

    #print(orderdata)
    #print(reader)
    dataset = Dataset.load_from_df(orderdata[['User ID', 'uniq_id', 'User rating for the product']], reader)
    trainset = dataset.build_full_trainset()

    svd = SVD()
    svd.fit(trainset)

    # Filter unrated products
    user_ratings = orderdata[orderdata['User ID'] == user_id]
    user_rated_products = user_ratings['uniq_id'].values
    all_products = orderdata['uniq_id'].unique()
    unrated_products = [prod for prod in all_products if prod not in user_rated_products]

    
    predictions = []
    for product_id in unrated_products:
        pred = svd.predict(user_id, product_id)
        predictions.append((product_id, pred.est))

    sorted_predictions = sorted(predictions, key=lambda x: x[1], reverse=True)

    recommended_product_ids = [prod[0] for prod in sorted_predictions[:n_recommendations]]
    
    # Get product information and include predicted ratings
    recommendations = catalogue[catalogue['uniq_id'].isin(recommended_product_ids)].copy()
    #print(recommendations)
    for idx, row in recommendations.iterrows():
        prediction = next((pred[1] for pred in predictions if pred[0] == row['uniq_id']), None)
        recommendations.loc[idx, 'predicted_rating'] = prediction
    recommendations_df = recommendations[['uniq_id', 'product_name', 'description', 'product_category_tree', 'predicted_rating', 'retail_price']]
    
    return recommendations_df
'''
# To test
#user_id = 'U06610'
#recommendations = svd_recommend_surprise(user_id, orderdata)
#print(recommendations)
