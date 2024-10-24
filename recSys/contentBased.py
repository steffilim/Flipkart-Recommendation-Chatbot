
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
