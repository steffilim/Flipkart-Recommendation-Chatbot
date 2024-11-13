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

'''
def load_data(table_name, n_rows):
    supabase = initialising_supabase()

    all_data = []
    batch_size = 1000  # Maximum rows Supabase allows per request
    start = 0

    while start < n_rows:
        # Fetch a batch of data
        response = supabase.table(table_name).select("*").range(start, start + batch_size - 1).execute()
        
        # If there's no data, break out of the loop
        if not response.data:
            break

        # Append the batch to the all_data list
        all_data.extend(response.data)
        start += batch_size  # Move to the next batch

        # print(f"Loaded {len(all_data)} rows so far...")  # Monitor loading progress

    # Convert the combined data to a DataFrame
    catalogue = pd.DataFrame(all_data)

    # converting overall_rating to numeric
    # catalogue['overall_rating'] = pd.to_numeric(database['overall_rating'], errors='coerce')

    catalogue['content'] = catalogue['description'].astype(str) + catalogue['product_specifications'].astype(str)

    # print("Shape of catalogue:", catalogue.shape)  # Check final shape after combining batches
    # print("Successfully loaded 'flipkart_cleaned' from Supabase")
    return catalogue

# Load the data
# df = load_product_data()
# print("Final shape:", df.shape)
'''

# Load the Sentence Transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
# model.save("sentenceTransformer")

''' computing embeddings '''
# precompute product embeddings
def precompute_product_embeddings(df, batch_size=1000, print_interval=100):
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
    # Fetch embeddings for the given list of product_ids from Supabase
    supabase = initialising_supabase()

    for product_id in product_ids:
        response = supabase.table("product_embeddings") \
                           .select("product_id, embedding_list") \
                           .eq("product_id", product_id) \
                           .execute()
    
    # Check if the response has data
    df = pd.DataFrame(columns=["product_id", "embedding_list"])

    #print(response.data)

    if response.data:
        data = response.data
        
        # Convert response data to DataFrame
        df = pd.DataFrame(data)
        # print('in get_product_embeddings function: matching product ids found, returning')
    else:
        # If no data is returned, return an empty DataFrame with the expected columns
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

''' filter function '''
# Function to Filter Products based on keywords - for features that have a 'hard' limit

def filter_products(product_name=None, price_limit=None, brand=None, overall_rating=None, product_specifications=None):
    # Build the SQL query dynamically based on the filters provided
    query = supabase.table("flipkart_cleaned").select("*")
    
    if product_name and isinstance(product_name, str) and product_name.strip():
        query = query.ilike("product_name", f"%{product_name}%")
    
    if price_limit is not None:
        query = query.lte("retail_price", price_limit)
    
    if brand and isinstance(brand, str) and brand.strip():
        query = query.ilike("brand", f"%{brand}%")
    
    if overall_rating is not None:
        query = query.gte("overall_rating", overall_rating)
    
    if product_specifications and isinstance(product_specifications, str) and product_specifications.strip():
        query = query.ilike("product_specifications", f"%{product_specifications}%")
    
    # Execute the query and get the results
    response = query.execute()
    return response.data

def recommend_top_products(user_query, filtered_products, top_n=10):
    supabase = initialising_supabase()
    '''
    user_query: string e.g. "i want a red shoe"
    # product_embeddings: embeddings of filtered products
    filtered_products: list of dictionary
    top_n = top n recommendations
    '''
    '''
    1. get a list of product ids from the filtered products
    2. use that list to get product embeddings
    3. calculate cosine similarity of user query with product embeddings obtained
    4. return top_n most similar 
    '''
    # Generate embedding for the user query
    # query_embedding = model.encode(user_query, convert_to_tensor=True)

    query_embedding = model.encode([user_query])[0]  
    # print("query embedding length ", len(query_embedding))
    # print("query embedding ", query_embedding)

    # Move query embedding to CPU
    # query_embedding = query_embedding.cpu()

    # Get product IDs from filtered products
    product_ids = [product['uniq_id'] for product in filtered_products]
    # print("product_ids ", product_ids)

    # Get specific product embeddings
    raw_product_embeddings = get_product_embeddings(product_ids)
    print("line 276: ", raw_product_embeddings)

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
            'product_id': product['uniq_id'],
            'product_name': product.get('product_name', 'N/A'),
            'similarity_score': similarity_score,
            'price': product.get('retail_price', 'N/A'),
            'description': product.get('description', 'N/A')
        })

    # Convert to DataFrame
    df = pd.DataFrame(top_products_with_scores)

    return df

''' TEST '''
'''
# Example extracted information from user input
extracted_info = {
    "product_name": "cycling shorts",
    "price_limit": 3000,
    "brand": "alisha",
    "overall_rating": '0',
    "product_specifications": ''
}

# filtering of products

# Start the timer
# start_time = time.time()

filtered_products = filter_products("flipkart_cleaned", product_name="cycling shorts", price_limit=3000, brand="alisha", overall_rating='0',product_specifications='')

# filter_time = time.time()
# Calculate the elapsed time
# filtered_time = filter_time - start_time
# print(f"Time taken to run the filter function: {filtered_time:.4f} seconds")

# getting recommendations
user_query = "i want red shoes"
top_products = recommend_top_products(user_query, filtered_products)
# End the timer
# end_time = time.time()

# Calculate the elapsed time
# elapsed_time = end_time - filter_time
# print(f"Time taken to run the recommendation function: {elapsed_time:.4f} seconds")
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