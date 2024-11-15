import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from pymongo import MongoClient
import pymongo
from supabase import create_client
from functions import initialising_supabase
### Targeted at new customers, built on demographic filtering to resolve the cold start problem

def load_order_data():
    supabase = initialising_supabase()
    # Load data from the flipkart_cleaned table in supabase
    order_data = pd.DataFrame(supabase.table('synthetic_v2_2k').select('*').execute().data)

    print("Successfully loaded order from Supabase")
 
    return order_data

def load_product_data():
    supabase = initialising_supabase()
    # Load data from the flipkart_cleaned table in supabase
    catalogue_data = pd.DataFrame(supabase.table('flipkart_cleaned_2k').select('*').execute().data)
    # Create the 'content' column by concatenating 'description'
    catalogue_data['content'] = catalogue_data['description'].astype(str) + ' ' + catalogue_data['product_specifications'].astype(str)
    # Ensure there are no NaN values which can cause issues
    catalogue_data['content'] = catalogue_data['content']. fillna("")
    print("Successfully loaded product data from Supabase")
    return catalogue_data

def get_trending_products():
    # supabase = initialising_supabase()
    # Fetch the catalogue & order data from Supabase
    catalogue = load_product_data()
    orderdata = load_order_data()   
    # Filter for purchases within the past year
    one_year_ago = datetime.now() - timedelta(days=365)
    recent_purchases = orderdata[pd.to_datetime(orderdata['Order Date']) >= one_year_ago]
    # Identify trending products by sale volume
    popular_products = recent_purchases.groupby('uniq_id')['User rating for the product'].count().reset_index()
    top_5_most_popular = popular_products.sort_values('User rating for the product', ascending=False).head()

    # Add product details from the catalog
    detailed_top_products = pd.merge(top_5_most_popular, catalogue, on='uniq_id', how='left')
    detailed_top_products = detailed_top_products[['uniq_id', 'product_name', 'discounted_price', 'description', 'User rating for the product']]
    
    print(detailed_top_products)
    return detailed_top_products

# get_trending_products()

# # finding the product name and saving it into the top_5_most_popular dataframe
# detailed_top_products = pd.merge(top_5_most_popular, catalog, on='Product ID', how='left')
# detailed_top_products = detailed_top_products[['Product ID','product_name', 'discounted_price', 'description', 'User rating for the product']]

# #print(detailed_top_products)

# # saving to dataframe for easy retrieval
# load_dotenv()
# MONGODB_URI = os.getenv("MONGODB_URI")
# FLIPKART = os.getenv("FLIPKART")


# client = pymongo.MongoClient(MONGODB_URI)
# mydb = client[FLIPKART]
# top5 = mydb.Top5Products

# # saving the top 5 most popular products to the database
# for index, row in detailed_top_products.iterrows():
#     document = {
#         "Product ID": row['Product ID'],
#         "product_name": row['product_name'],
#         "discounted_price": row['discounted_price'],
#         "description": row['description'],
#         "User rating for the product": row['User rating for the product']
#     }
#     #print(document)
#     top5.insert_one(document)
