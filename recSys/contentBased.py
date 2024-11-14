import sys
import os

import time
import numpy as np
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
from supabase import create_client, Client
# from weighted import filter_products, fetch_filtered_products
# from functions import initialising_supabase, load_product_data

# lsa_matrix = load(lsa_matrix_file)

# supabase = initialising_supabase()
def initialising_supabase():
    load_dotenv()
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_API_KEY = os.getenv("SUPABASE_API_KEY")
    supabase = create_client(SUPABASE_URL, SUPABASE_API_KEY)
    return supabase

''' loading product data '''
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

# Load the Sentence Transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
# model.save("sentenceTransformer")

''' computing embeddings '''
# precompute product embeddings
def precompute_product_embeddings(df, batch_size=1000, print_interval=100):
    """
    Precomputes embeddings for product descriptions using a sentence transformer model.
    
    Args:
        df (pd.DataFrame): DataFrame containing product information with 'content' and 'uniq_id' columns
        batch_size (int): Number of products to process in each batch
        print_interval (int): Interval for printing progress updates
    
    Returns:
        list: List of dictionaries containing product IDs and their corresponding embeddings
    """
    print("Starting precompute_product_embeddings")

    product_descriptions = df['content'].tolist()
    product_ids = df['uniq_id'].tolist()
    total_products = len(product_descriptions)
    
    product_embeddings = []
    
    # Process embeddings in batches
    for start in range(0, total_products, batch_size):
        end = min(start + batch_size, total_products)
        batch_descriptions = product_descriptions[start:end]
        batch_ids = product_ids[start:end]
        
        # Compute embeddings for the batch
        with torch.no_grad():  # Avoid computing gradients to save memory
            batch_embeddings = model.encode(batch_descriptions, convert_to_tensor=True)
        
        # Store embeddings with their associated product IDs
        product_embeddings.extend(
            {"product_id": pid, "embedding": embedding} for pid, embedding in zip(batch_ids, batch_embeddings)
        )
        
        # Print progress for every print_interval items processed
        if (end % print_interval == 0) or (end == total_products):
            print(f"Processed {end}/{total_products} products...")

        print("precompute_product_embeddings end")

    return product_embeddings

# storing on local drive
# def store_embeddings(product_embeddings):
#     np.save("product_embeddings.npy", product_embeddings.cpu().numpy())

# loading from local drive
# def load_embeddings(product_embeddings_file):
#     product_embeddings_file = torch.tensor(np.load("product_embeddings.npy"))

''' storing data - into supabase (wip)'''
def store_product_embeddings_in_supabase(product_embeddings):
    for entry in product_embeddings:
        pid = entry['product_id']
        embedding = entry['embedding']
        print('pid', pid)

        # Convert embedding tensor to a list of floats if it's a tensor
        embedding_list = embedding.cpu().numpy().tolist() if isinstance(embedding, torch.Tensor) else embedding
        
        print("embedding_list", embedding_list)

        # Convert embedding list to a string (comma-separated values)
        embedding_str = ','.join(map(str, embedding_list))

        # Insert product_id and corresponding embedding string into Supabase
        response = supabase.table("embeddings").upsert({
            "product_id": pid,
            "embedding": embedding_str  # Storing the embedding as a string
        }).execute()

        # Debug: Print the response from Supabase
        print(f"Response from Supabase: {response}")
    
    print("Finished storing product embeddings in Supabase.")

''' retrieving embeddings for a list of product_ids'''
def get_product_embeddings(product_ids):
    print("product_ids", product_ids)
    '''
    Fetch embeddings for the given list of product_ids from Supabase   

    Args:
        product_ids (list of string): List of product IDs to fetch embeddings for.
                                    Product IDs cannot be empty strings.
    Returns:
        pandas.DataFrame: DataFrame containing product_id and embedding_list columns.
        Returns empty DataFrame if no matches found or if input list is empty.
    Raises:
        TypeError: If product_ids is not a list
        ValueError: If any product_id in the list is not a string or is an empty string
    '''
    supabase = initialising_supabase()
    
    # Input validation
    if not isinstance(product_ids, list):
        raise TypeError("product_ids must be a list")

    # Return empty DataFrame immediately if empty list
    if not product_ids:
        print("Empty input list provided")
        return pd.DataFrame(columns=["product_id", "embedding_list"])
    
    # Validate all items are strings
    if not all(isinstance(pid, str) for pid in product_ids):
        raise ValueError("All product IDs must be strings")
    
    # Initialize response outside the loop
    response = None
    
    '''
    for product_id in product_ids:
        response = supabase.table("product_embeddings") \
                           .select("product_id, embedding_list") \
                           .eq("product_id", product_id) \
                           .execute()
    '''

    response = supabase.table("product_embeddings") \
                      .select("product_id, embedding_list") \
                      .in_("product_id", product_ids) \
                      .execute()
        
    df = pd.DataFrame(columns=["product_id", "embedding_list"])

    if response and response.data:
        df = pd.DataFrame(response.data)
    else:
        print('no matching data found in product embeddings')
    
    return df

''' convertion of embeddings list 'text' to 'float' (cannot save as array of 32 bits in supabase)'''
# remove brackets from embedding list
def remove_brackets_from_embedding_list(df):
    if 'embedding_list' in df.columns:
        # print('Removing brackets from embedding_list')
        
        # Remove brackets from each entry in embedding_list
        df['embedding_list'] = df['embedding_list'].str.replace('[', '', regex=False).str.replace(']', '', regex=False)
    
    return df

# convertion from list to float
def convert_embedding_list_to_floats(df):
    if 'embedding_list' in df.columns:
        # print('Converting embedding_list to a list of floats')
        
        # Convert the comma-separated string to a list of floats
        df['embedding_list'] = df['embedding_list'].apply(
            lambda x: list(map(float, x.split(',')))
        )
    
    return df

# checking the type of elemenets
def check_if_embedding_list_is_float(df):
    # Check if embedding_list is a list of floats
    if 'embedding_list' in df.columns:
        df['is_float_list'] = df['embedding_list'].apply(
            lambda x: isinstance(x, list) and all(isinstance(i, float) for i in x)
        )
    return df

''' recommendation function '''
def recommend_top_products(user_query, filtered_products, top_n=10):
    '''
    Recommend top N products based on similarity to the user's query.

    Args:
        user_query (str): The query input by the user, e.g., "I want a red shoe".
                          Should be a non-empty string.
        filtered_products (list of dict): List of dictionaries, where each dictionary contains 
                                          product information (e.g., 'uniq_id', 'product_name', 
                                          'retail_price', 'description'). Each dictionary must 
                                          contain a valid 'uniq_id' as a unique identifier.
        top_n (int, optional): Number of top recommended products to return. Default is 10.

    Returns:
        pandas.DataFrame: DataFrame containing columns 'product_id', 'product_name', 
                          'similarity_score', 'price', and 'description' for the top N recommended 
                          products based on similarity to the user query.

    Raises:
        ValueError: If user_query is not a string or is empty.
        TypeError: If filtered_products is not a list of dictionaries, or if any dictionary 
                   does not contain a 'uniq_id'.
    '''

    supabase = initialising_supabase()

    query_embedding = model.encode([user_query])[0]  

    # Get product IDs from filtered products
    product_ids = [product['uniq_id'] for product in filtered_products]
    # print("product_ids ", product_ids)

    # Get specific product embeddings
    raw_product_embeddings = get_product_embeddings(product_ids)
    print("line 260: ", raw_product_embeddings)

    # Converting it to float instead of string
    text_product_embeddings = remove_brackets_from_embedding_list(raw_product_embeddings)
    product_embeddings = convert_embedding_list_to_floats(text_product_embeddings)
    product_embeddings = product_embeddings['embedding_list']

    # print(product_embeddings)

    # Flatten the embedding list into a single 1D array
    product_embeddings_flat = [embedding for sublist in product_embeddings for embedding in sublist]

    # Calculate similarity scores with precomputed product embeddings
    product_embeddings_array = np.array(product_embeddings_flat)

    # Reshaping Array
    product_embeddings_array = product_embeddings_array.reshape(-1, 384)

    # Calculate similarity scores
    similarities = cosine_similarity([query_embedding], product_embeddings_array)[0]
    
    # Get top N most similar products
    sorted_indices = np.argsort(similarities)[::-1][:top_n]
     # Prepare the top products with their similarity scores
    top_products_with_scores = []
    for idx in sorted_indices:
        product = filtered_products[idx]
        similarity_score = similarities[idx]
        top_products_with_scores.append({
            'uniq_id': product['uniq_id'],
            'similarity_score': similarity_score,
        })

    # Convert to DataFrame
    df = pd.DataFrame(top_products_with_scores)

    return df

''' TEST '''
'''
# Example extracted information from user input
extracted_info = {
    "product_name": "cycling shorts",
    "max_price": 3000,
    "brand": "alisha",
    "specifications": "No preference"
}

# filtering of products

# Start the timer
start_time = time.time()

filtered_products = fetch_filtered_products(extracted_info)

top_products = recommend_top_products(extracted_info, filtered_products)
# End the timer
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - filter_time
print(f"Time taken to run the recommendation function: {elapsed_time:.4f} seconds")
top_products
'''


















'''

# lsa_matrix = load(lsa_matrix_file)

"""load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI")
FLIPKART = os.getenv("FLIPKART")

client = pymongo.MongoClient(MONGODB_URI)
flipkart = client[FLIPKART]

product_data_file = flipkart.catalogue
lsa_matrix_file = 'lsa_matrix.joblib'
"""
# Function to load and preprocess the data
def load_product_data(product_data_file):
    cursor = product_data_file.find({})
    catalogue = pd.DataFrame(list(cursor))  
    catalogue['content'] = (catalogue['product_name'].astype(str) + ' ' +
                     catalogue['product_category_tree'].astype(str) + ' ' +
                     catalogue['retail_price'].astype(str) + ' ' +
                     catalogue['discounted_price'].astype(str) + ' ' +
                     catalogue['discount'].astype(str) + ' ' +
                     catalogue['description'].astype(str) + ' ' +
                     catalogue['overall_rating'].astype(str) + ' ' +
                     catalogue['brand'].astype(str) + ' ' +
                     catalogue['product_specifications'].astype(str))
    
    catalogue['content'] = catalogue['content'].fillna('')  # Ensure there are no NaN values which can cause issues

    print("Successfully loaded DataFrame from MongoDB")
    return catalogue


# Function to calculate or load the LSA matrix
def get_lsa_matrix(catalogue_df, catalogue_db, lsa_matrix_file):
    recalculate_lsa = False
    if os.path.exists(lsa_matrix_file):
        lsa_matrix_mtime = os.path.getmtime(lsa_matrix_file)
        print(lsa_matrix_mtime)

        # retrieving the latest modification time from the database
        latest_update = catalogue_db.find_one(sort=[("modified_time", pymongo.DESCENDING)])['modified_time']
        catalogue_mtime = latest_update.timestamp()
        if catalogue_mtime > lsa_matrix_mtime:
            print("product database changed... recalculating lsa matrix")
            recalculate_lsa = True
    else:
        print("no lsa matrix found... calculating lsa matrix")
        recalculate_lsa = True

    if recalculate_lsa:
        print("commencing calculating lsa")
        vectorizer = CountVectorizer()
        bow = vectorizer.fit_transform(catalogue_df['content'])

        tfidf_transformer = TfidfTransformer()
        tfidf = tfidf_transformer.fit_transform(bow)

        lsa = TruncatedSVD(n_components=100, algorithm='arpack')
        lsa.fit(tfidf)
        lsa_matrix = lsa.transform(tfidf)
        dump(lsa_matrix, lsa_matrix_file)
    else:
        print("lsa matrix found... loading...")
        lsa_matrix = load(lsa_matrix_file)
   #  print("LSA Matrix: ", lsa_matrix)
    return lsa_matrix

# Function to get recommendations
def get_recommendations(item, catalogue, lsa_matrix):
    print("content")
    cursor = catalogue.find({})
    catalogue = pd.DataFrame(list(cursor))  
    match = process.extractOne(item, catalogue['product_name'])
    closest_match = match[0]
    score = match[1]

    if score < 70:
        return "No close match found"
    
    product_index = catalogue[catalogue['product_name'] == closest_match].index[0]
    similarity_scores = cosine_similarity(lsa_matrix[product_index].reshape(1, -1), lsa_matrix)
    
    similar_products = list(enumerate(similarity_scores[0]))
    sorted_similar_products = sorted(similar_products, key=lambda x: x[1], reverse=True)[1:10]

    recommendations = []
    for i, score in sorted_similar_products:
        # recommendations.append(catalogue.loc[i, 'product_name'])
        product_name = catalogue.loc[i, 'product_name']
        retail_price = catalogue.loc[i, 'retail_price']
        description = catalogue.loc[i, 'description']
        overall_rating = catalogue.loc[i, 'overall_rating']
        recommendations.append({
            'product_name': product_name,
            'retail_price': retail_price,
            'description': description,
            'overall_rating': overall_rating,
            'similarity_score': score
        })
    
    # print(recommendations)
    return recommendations


#user_product = 'socks'
#recommendations = get_recommendations(user_product, product_data, lsa_matrix)

'''