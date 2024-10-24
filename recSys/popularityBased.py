import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from pymongo import MongoClient
import pymongo
### Targeted at new customers, built on demographic filtering to resolve the cold start problem

# Loading the dataset
df = pd.read_csv('newData/synthetic_v2.csv')
df['Order Date'] = pd.to_datetime(df['Order Date'])

catalog = pd.read_csv('newData/flipkart_cleaned.csv')
catalog.rename(columns={'uniq_id': 'Product ID'}, inplace=True)


# Get purchases within the past year
one_year_ago = datetime.now() - timedelta(days=365)
recent_purchases = df[df['Order Date'] >= one_year_ago]

# Getting trending products (by sale volume)
popular_products = pd.DataFrame(recent_purchases.groupby('Product ID')['User rating for the product'].count())
top_5_most_popular = popular_products.sort_values('User rating for the product', ascending=False).head()

# finding the product name and saving it into the top_5_most_popular dataframe
detailed_top_products = pd.merge(top_5_most_popular, catalog, on='Product ID', how='left')
detailed_top_products = detailed_top_products[['Product ID','product_name', 'discounted_price', 'description', 'User rating for the product']]

#print(detailed_top_products)

# saving to dataframe for easy retrieval
load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI")
FLIPKART = os.getenv("FLIPKART")


client = pymongo.MongoClient(MONGODB_URI)
mydb = client[FLIPKART]
top5 = mydb.Top5Products

# saving the top 5 most popular products to the database
for index, row in detailed_top_products.iterrows():
    document = {
        "Product ID": row['Product ID'],
        "product_name": row['product_name'],
        "discounted_price": row['discounted_price'],
        "description": row['description'],
        "User rating for the product": row['User rating for the product']
    }
    #print(document)
    top5.insert_one(document)
