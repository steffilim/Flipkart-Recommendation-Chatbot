
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

"""load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI")
FLIPKART = os.getenv("FLIPKART")

client = pymongo.MongoClient(MONGODB_URI)
flipkart = client[FLIPKART]

product_data_file = flipkart.catalogue
lsa_matrix_file = 'lsa_matrix.joblib'
"""

''' 
# Function to load and preprocess the data
def load_product_data(product_data_file):
    cursor = product_data_file.find({})
    catalogue = pd.DataFrame(list(cursor))  
    # print(catalogue)
    
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
'''

# df = load_product_data(flipkart.catalogue)
# Save the DataFrame to a CSV file
# df.to_csv("newData/flipkart_cleaned.csv", index=False)

# Function to Filter Products based on keywords - for features that have a 'hard' limit
def filter_products(database, product_name=None, price_limit=None, brand=None, overall_rating=None):
   
    filtered_df = database.copy()

    # Filter by product name
    if product_name and isinstance(product_name, str) and product_name.strip():
        # Check for NaN values in the 'product_name' column and filter
        filtered_df = filtered_df[filtered_df['product_name'].notna() & 
                                   filtered_df['product_name'].str.contains(product_name, case=False)]
        
    # Filter by retail price
    if price_limit is not None:  # Check if price_limit is explicitly set
        filtered_df = filtered_df[filtered_df['retail_price'].notna() & 
                                   (filtered_df['retail_price'] <= price_limit)]
        
    # Filter by brand
    if brand and isinstance(brand, str) and brand.strip():  # Check if brand is a non-empty string
        # Check for NaN values in the 'brand' column and filter
        filtered_df = filtered_df[filtered_df['brand'].notna() & 
                                   filtered_df['brand'].str.contains(brand, case=False)]
 
    # Filter by minimum overall rating
    if overall_rating is not None:  # Check if overall_rating is explicitly set
        filtered_df = filtered_df[filtered_df['overall_rating'].notna() & 
                                   (filtered_df['overall_rating'] >= overall_rating)]

    return filtered_df

# Example extracted information from user input
extracted_info = {
    "product_name": "cycling shorts",
    "price_limit": 3000,
    "brand": "alisha",
    "overall_rating": 0,
}

'''
# Apply filters based on extracted information
result = filter_products(database, **extracted_info)
print(result)
'''

# Load the Sentence Transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
# model.save("sentenceTransformer")

# precompute product embeddings
def precompute_product_embeddings(df):
    product_descriptions = df['content'].tolist()
    product_embeddings = model.encode(product_descriptions, convert_to_tensor=True)
    return product_embeddings

def store_embeddings(product_embeddings):
    np.save("product_embeddings.npy", product_embeddings.cpu().numpy())

def load_embeddings(product_embeddings_file):
    product_embeddings_file = torch.tensor(np.load("product_embeddings.npy"))

# product_embeddings = precompute_product_embeddings(df)

def recommend_top_products(user_query, product_embeddings, df, top_n=20):
    # Generate embedding for the user query
    query_embedding = model.encode(user_query, convert_to_tensor=True)

    # Move query embedding to CPU
    query_embedding = query_embedding.cpu()
    
    # Calculate similarity scores with precomputed product embeddings
    similarities = util.cos_sim(query_embedding, product_embeddings).squeeze()
    
    # Get top N most similar products
    top_indices = similarities.argsort(descending=True)[:top_n]
    top_products = df.iloc[top_indices]
    
    return top_products

'''
# Generate precomputed embeddings
product_embeddings = precompute_product_embeddings(filtered_df)

# User query example
user_query = "I want shorts for cycling"

# Call the recommendation function with precomputed embeddings
top_recommendations = recommend_top_products(user_query, product_embeddings, filtered_df, top_n=3)

# Display the recommended products
print(top_recommendations[['product_name', 'description']])
'''

'''
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

### example
# product_data = load_product_data(product_data_file)
# lsa_matrix = get_lsa_matrix(product_data, lsa_matrix_file)

#user_product = 'socks'
#recommendations = get_recommendations(user_product, product_data, lsa_matrix)
'''