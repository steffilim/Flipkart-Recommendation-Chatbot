import os
import time
from joblib import dump, load
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process

product_data_file = 'newData/flipkart_cleaned.csv'
lsa_matrix_file = 'lsa_matrix.joblib'
# lsa_matrix = load(lsa_matrix_file)

# Function to load and preprocess the data
def load_product_data(product_data_file):
    df = pd.read_csv(product_data_file)
    df['content'] = (df['product_name'].astype(str) + ' ' + df['product_category_tree'].astype(str) + ' ' + df['retail_price'].astype(str) + ' ' + df['discounted_price'].astype(str) + ' ' + df['discount'].astype(str) + ' ' + df['description'].astype(str) + ' ' +
    df['overall_rating'].astype(str) + ' ' + df['brand'].astype(str) + ' ' + df['product_specifications'].astype(str))
    df['content'] = df['content'].fillna('')

    print("successfully loaded df")
    return df

# Function to calculate or load the LSA matrix
def get_lsa_matrix(df, lsa_matrix_file):
    recalculate_lsa = False
    if os.path.exists(lsa_matrix_file):
        lsa_matrix_mtime = os.path.getmtime(lsa_matrix_file)
        product_data_mtime = os.path.getmtime(product_data_file)
        if product_data_mtime > lsa_matrix_mtime:
            print("product databased changed... recalculating lsa matrix")
            recalculate_lsa = True
    else:
        print("no lsa matrix found... calculating lsa matrix")
        recalculate_lsa = True

    if recalculate_lsa:
        print("commencing calculating lsa")
        vectorizer = CountVectorizer()
        bow = vectorizer.fit_transform(df['content'])

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
def get_recommendations(user_product, df, lsa_matrix):
    match = process.extractOne(user_product, df['product_name'])
    closest_match = match[0]
    score = match[1]

    if score < 70:
        return "No close match found"
    
    product_index = df[df['product_name'] == closest_match].index[0]
    similarity_scores = cosine_similarity(lsa_matrix[product_index].reshape(1, -1), lsa_matrix)
    
    similar_products = list(enumerate(similarity_scores[0]))
    sorted_similar_products = sorted(similar_products, key=lambda x: x[1], reverse=True)[1:10]

    recommendations = []
    for i, score in sorted_similar_products:
        # recommendations.append(df.loc[i, 'product_name'])
        product_name = df.loc[i, 'product_name']
        retail_price = df.loc[i, 'retail_price']
        overall_rating = df.loc[i, 'overall_rating']
        recommendations.append({
            'product_name': product_name,
            'retail_price': retail_price,
            'overall_rating': overall_rating,
            'similarity_score': score
        })
    
    print(recommendations)
    return recommendations

product_data = load_product_data(product_data_file)
lsa_matrix = get_lsa_matrix(product_data, lsa_matrix_file)

user_product = 'socks'
recommendations = get_recommendations(user_product, product_data, lsa_matrix)
