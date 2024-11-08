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
    catalogue_data = pd.DataFrame(supabase.table('flipkart_cleaned').select('*').execute().data)

    # Create the 'content' column by concatenating 'description' and 'product_specifications'
    catalogue_data['content'] = catalogue_data['description'].astype(str) + ' ' + catalogue_data['product_specifications'].astype(str)
     # Ensure there are no NaN values which can cause issues
    catalogue_data['content'] = catalogue_data['content'].fillna('') 

    print("Successfully loaded product data from Supabase")
 
    return catalogue_data

def load_order_data():
    supabase = initialising_supabase()
    # Load data from the flipkart_cleaned table in supabase
    order_data = pd.DataFrame(supabase.table('synthetic_v2').select('*').execute().data)

    print("Successfully loaded order from Supabase")
 
    return order_data



def svd_recommend_surprise(user_id, catalogue, n_recommendations=20):
    print("SVD")
    supabase = initialising_supabase()
     # Fetch the catalogue & users data from Supabase
    catalogue = load_product_data()
    orderdata = load_order_data()

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

    # Get product information for the recommended product IDs
    recommendations = catalogue[catalogue['uniq_id'].isin(recommended_product_ids)].copy()

    # Add predicted ratings to the recommendations
    for idx, row in recommendations.iterrows():
        prediction = next((pred[1] for pred in predictions if pred[0] == row['uniq_id']), None)
        recommendations.loc[idx, 'predicted_rating'] = prediction

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
