import os
import time
from joblib import dump, load  # to store matrix
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process

# File paths for saving/loading 
lsa_matrix_file = 'lsa_matrix.joblib'
product_data_file = '/Users/justinlum/Downloads/NUS/Flipkart-Recommendation-Chatbot/newData/flipkart_cleaned.csv'

# Load product data
df = pd.read_csv(product_data_file)

# Combining product_name and features into a single string
df['content'] = df['product_name'].astype(str) + ' ' + df['product_category_tree'].astype(str) + ' ' + df['retail_price'].astype(str) + ' ' + df['discounted_price'].astype(str) + ' ' + df['discount'].astype(str) + ' ' + df['description'].astype(str) + ' ' + df['overall_rating'].astype(str) + ' ' + df['brand'].astype(str) + ' ' + df['product_specifications'].astype(str)

df['content'] = df['content'].fillna('')

# Load or compute LSA matrix
if os.path.exists(lsa_matrix_file):
    print("Loading LSA matrix from file...")
    lsa_matrix = load(lsa_matrix_file)
else:
    print("Computing LSA matrix...")
    vectorizer = CountVectorizer()
    bow = vectorizer.fit_transform(df['content'])

    tfidf_transformer = TfidfTransformer()
    tfidf = tfidf_transformer.fit_transform(bow)

    # Apply LSA
    lsa = TruncatedSVD(n_components=100, algorithm='arpack')
    lsa_matrix = lsa.fit_transform(tfidf)

    # Save the LSA matrix
    dump(lsa_matrix, lsa_matrix_file)
    print("LSA matrix saved.")

# Define the recommendation system as a function
def recommendation_system(user_product, top_n=10):
    # Use fuzzy matching to find the closest product name
    match = process.extractOne(user_product, df['product_name'])
    
    # If no match is found, return an empty list
    if not match:
        print(f"No match found for '{user_product}'")
        return []

    closest_match = match[0]
    score = match[1]

    # If the match score is less than 70, return an empty list
    if score < 70:
        print(f"Closest match '{closest_match}' has a low score: {score}")
        return []

    # Find the index of the closest product
    product_index = df[df['product_name'] == closest_match].index[0]

    # Compute cosine similarities using the LSA matrix
    similarity_scores = cosine_similarity(lsa_matrix[product_index].reshape(1, -1), lsa_matrix)

    # Get the top N most similar products
    similar_products = list(enumerate(similarity_scores[0]))
    sorted_similar_products = sorted(similar_products, key=lambda x: x[1], reverse=True)[1:top_n+1]

    # Return just the top N similar product names (for evaluation purposes)
    return [df.loc[i, 'product_name'] for i, _ in sorted_similar_products]



# Main block (if you want to use the script standalone)
if __name__ == "__main__":
    # Get user input
    user_product = input("Enter a product: ")

    # Start timer after user input
    start_time = time.time()

    # Call the recommendation system function
    recommendations = recommendation_system(user_product)

    # Print recommendations
    for rec in recommendations:
        print(rec)

    # End timer for the entire program
    end_time = time.time()
    print("Time taken to find recommendations: {:.2f} seconds".format(end_time - start_time))
