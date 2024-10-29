
import time
import numpy as np
import os
from dotenv import load_dotenv
from sklearn.feature_extraction.text import CountVectorizer,  TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from joblib import dump, load
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process
import pymongo
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

# lsa_matrix = load(lsa_matrix_file)

''' Load Environment Variables '''
load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI")
FLIPKART = os.getenv("FLIPKART")

client = pymongo.MongoClient(MONGODB_URI)
flipkart = client[FLIPKART]
product_collection = flipkart["catalogue"]
embeddings_collection = flipkart["productEmbeddings"]

# print(client.list_database_names())  # Lists all databases
# print(flipkart.list_collection_names())  # Lists all collections in the flipkart database

''' Preprocessing the Product Database '''
# Load the Sentence Transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Function to load and preprocess the data
# Returns: Dataframe of entire product database from MongoDB 
def load_product_data():
    cursor = product_collection.find({})
    catalogue = pd.DataFrame(list(cursor))  
    print("Successfully loaded DataFrame from MongoDB")
    return catalogue

# Function to compute product embeddings
def precompute_product_embeddings(df):
    product_descriptions = df['content'].tolist()
    product_embeddings = model.encode(product_descriptions, convert_to_tensor=True)
    return product_embeddings

# Function to precompute product embeddings and store in MongoDB
# Returns: product_embeddings
def precompute_and_store_product_embeddings(df):
    product_descriptions = df['content'].tolist()
    product_embeddings = model.encode(product_descriptions, convert_to_tensor=True)

    try:
        # Store embeddings in MongoDB
        embeddings_data = [
        {"product_id": row["uniq_id"], "embedding": embedding.cpu().numpy().tolist()}
        for row, embedding in zip(df.to_dict(orient="records"), product_embeddings)
    ]
        result = embeddings_collection.insert_many(embeddings_data)
        print(f"Successfully stored {len(result.inserted_ids)} embeddings in MongoDB")

    except pymongo.errors.PyMongoError as e:
        print("Error while inserting embeddings:", e)
    
    return product_embeddings

# Loading all embeddings from MongoDB
# Returns: Embeddings
def load_embeddings_from_mongo():
    embeddings_data = embeddings_collection.find({})
    embeddings = torch.tensor([entry["embedding"] for entry in embeddings_data])
    return embeddings

# Load embeddings only for filtered products
# Returns: Embeddings
def load_filtered_embeddings_from_mongo(filtered_ids):
    # Use product IDs to load embeddings for only filtered products
    embeddings_data = embeddings_collection.find({"product_id": {"$in": filtered_ids}})
    embeddings = torch.tensor([entry["embedding"] for entry in embeddings_data if entry["product_id"] in filtered_ids])

    return embeddings

''' Content Based Recommender '''
# Function to query MongoDB with keywords
# Returns: Dataframe of filtered products
def query_catalogue_with_keywords(product_name=None, price_limit=None, brand=None, overall_rating=None):
    query = {}
    
    # Product name filter
    if product_name:
        query['product_name'] = {"$regex": product_name, "$options": "i"}
        
    # Price limit filter
    if price_limit is not None:
        query['retail_price'] = {"$lte": price_limit}
        
    # Brand filter
    if brand:
        query['brand'] = {"$regex": brand, "$options": "i"}
    
    # Overall rating filter
    if overall_rating is not None:
        query['overall_rating'] = {"$gte": overall_rating}
    
    # Execute query
    filtered_data = pd.DataFrame(list(product_collection.find(query)))
    print("Successfully filtered data from MongoDB")
    return filtered_data

# Recommendation function
# Returns top 10 products
def recommend_top_products(user_query, product_embeddings, df, top_n=20):
    
    # Generate embedding for the user query
    query_embedding = model.encode(user_query, convert_to_tensor=True)

    # Move query embedding to CPU
    query_embedding = query_embedding.cpu()
    
    # Calculate similarity scores with precomputed product embeddings
    similarities = util.cos_sim(query_embedding, product_embeddings).squeeze()
    
    # Get top N most similar products
    top_indices = similarities.argsort(descending=True)[:top_n]

    # Handle case where no valid recommendations are found
    if len(top_indices) == 0:
        return {"message": "No recommendations found."}
    
    top_products = df.iloc[top_indices]

    top_products = df.iloc[top_indices]
    
    return top_products

''' Precomputing Example '''
'''
# Load product data
start_time = time.time()

load_start_time = time.time()
df = load_product_data()
print(f"Time taken to load data: {time.time() - load_start_time:.2f} seconds")

# Precompute and store all embeddings
store_embeddings_start_time = time.time()
product_embeddings = precompute_and_store_product_embeddings(df)
print(f"Time taken to store data: {time.time() - store_embeddings_start_time:.2f} seconds")
'''

''' Recommender Example '''
start_time = time.time()


# User Query
user_query = "i want a red sofa"

# Filtering Data
filter_start_time = time.time()
filtered_df = query_catalogue_with_keywords(product_name='sofa')
print(f"Time taken for filtering data: {time.time() - filter_start_time:.2f} seconds")

if len(filtered_df) > 1000:
    filtered_df = filtered_df.sample(n=1000, random_state=42).reset_index(drop=True)

# Extract the filtered product IDs
id_extraction_start_time = time.time()
filtered_product_ids = filtered_df["uniq_id"].astype(str).tolist()
print(f"Time taken for extracting filtered product IDs: {time.time() - id_extraction_start_time:.2f} seconds")

# Load only the embeddings for the filtered products
load_embeddings_start_time = time.time()
filtered_product_embeddings = load_filtered_embeddings_from_mongo(filtered_product_ids)
print(f"Time taken for loading filtered embeddings: {time.time() - load_embeddings_start_time:.2f} seconds")

# Checks
print(f"Number of rows in filtered_df: {len(filtered_df)}")
print(f"Number of embeddings: {filtered_product_embeddings.shape[0]}")

# Get recommendations
recommendation_start_time = time.time()
top_recommendations = recommend_top_products(user_query, filtered_product_embeddings, filtered_df)
print(f"Time taken for generating recommendations: {time.time() - recommendation_start_time:.2f} seconds")

print(top_recommendations[['product_name', 'description']])


total_time = time.time() - start_time
print(f"Total time taken for the example run: {total_time:.2f} seconds")