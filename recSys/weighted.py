from collaborative import svd_recommend_surprise, svd_recommend_surprise_filtered

from contentBased import recommend_top_products 

import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
from dotenv import load_dotenv
import os
import pymongo
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from supabase import create_client
# from functions import initialising_supabase, load_product_data
from sklearn.preprocessing import MinMaxScaler 
import time

#MONGODB_URI = os.getenv("MONGODB_URI")
#FLIPKART = os.getenv("FLIPKART")

#client = pymongo.MongoClient(MONGODB_URI)
#flipkart = client[FLIPKART]

#product_data_file = flipkart.catalogue
# supabase = initialising_supabase()
# lsa_matrix_file = 'lsa_matrix.joblib'

''' loading supabase'''
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

    # print("Successfully loaded product data from Supabase")
 
    return catalogue_data

def load_order_data():
    supabase = initialising_supabase()
    # Load data from the flipkart_cleaned table in supabase
    order_data = pd.DataFrame(supabase.table('synthetic_v2').select('*').execute().data)

    # sfully loaded order from Supabase")
 
    return order_data

''' filter products '''
def filter_products(product_name=None, price_limit=None, brand=None, product_specifications=None):
    # Build the SQL query dynamically based on the filters provided
    supabase = initialising_supabase()
    query = supabase.table("flipkart_cleaned").select("*")
    
    if product_name and isinstance(product_name, str) and product_name.strip():
        query = query.ilike("product_name", f"%{product_name}%")
    
    if price_limit is not None:
        query = query.lte("retail_price", price_limit)
    
    if brand and isinstance(brand, str) and brand.strip():
        query = query.ilike("brand", f"%{brand}%")
    
    if product_specifications and isinstance(product_specifications, str) and product_specifications.strip():
        query = query.ilike("product_specifications", f"%{product_specifications}%")
    
    # Execute the query and get the results
    response = query.execute()
    return response.data

def fetch_filtered_products(extracted_info):
    # ensure correct inputs
    # ensure correct type
    # if cannot find in dictionary, just none
    # Extract values from the dictionary with default values for any missing keys
    product_name = extracted_info.get("Product Item")
    price_limit = extracted_info.get("Budget")
    brand = extracted_info.get("Brand")
    product_specifications = extracted_info.get("Product Details")

    # print(product_name, price_limit, brand, product_specifications)

    # Convert "Not specified" to None for each field
    if product_name == "No preference":
        product_name = None
    if price_limit == "No preference":
        price_limit = None
    if brand == "No preference":
        brand = None
    if product_specifications == "No preference":
        product_specifications = None

    # Print each field and its type after processing
    '''
    print("product_name:", product_name, "| Type:", type(product_name))
    print("price_limit:", price_limit, "| Type:", type(price_limit))
    print("brand:", brand, "| Type:", type(brand))
    print("overall_rating:", overall_rating, "| Type:", type(overall_rating))
    print("product_specifications:", product_specifications, "| Type:", type(product_specifications))
    '''

    if price_limit is not None and price_limit != "Not specified":
        try:
            price_limit = float(price_limit)
        except ValueError:
            print("Warning: price_limit is not a valid number:", price_limit)
            price_limit = None

    # print(product_name, price_limit, brand, product_specifications)

    filtered_products = filter_products(
        product_name=product_name,
        price_limit=price_limit,
        brand=brand,
        product_specifications=product_specifications
    )

    return filtered_products

'''
fetch_filtered_products( {'Related to Follow-Up Questions': 'New', 'Available in Store': 'Yes', 'Brand': 'Alisha', 'Product Item': 'Cycling shorts', 'Product Details': 'Not specified', 'Budget': 'Not specified', 'Fields Incompleted': '2', 'To-Follow-Up': 'Yes', 'Follow-Up Question': 'Could you please specify any specific features or specifications you need for the cycling shorts? Do you have a budget range in mind for this purchase?'})
'''

''' helper functions '''
def get_user_query(extracted_info, keys=['product_name', 'brand', 'specifications'], separator=', '):
    result = ''
    
    # Iterate over the list of keys to extract values
    for key in keys:
        # Check if key exists and its value is not 'No preference'
        if key in extracted_info and extracted_info[key] != 'No preference':
            result += str(extracted_info[key]) + separator
    
    # Remove the trailing separator
    result = result.rstrip(separator)
    return result

# fetching only specified columns: 'uniq_id, product_name, brand, retail_price, discounted_price, description'
def get_product_details_from_supabase(uniq_ids, columns = None):
    """
    Fetch full product details from Supabase based on provided unique IDs and selected columns.

    Args:
        uniq_ids (list of str): List of unique product IDs to fetch details for.
                                If empty, an empty DataFrame is returned.
        columns (list of str, optional): List of columns to select from the database.
                                         Defaults to common product detail columns.

    Returns:
        pandas.DataFrame: DataFrame containing the requested product details.
                          Returns an empty DataFrame if `uniq_ids` is empty.
    
    Raises:
        TypeError: If `uniq_ids` is not a list.
        ValueError: If any ID in `uniq_ids` is not a string or if `columns` is not a list.
    """
    print("get product details from supabase")
    # Ensure uniq_ids is a list
    if not isinstance(uniq_ids, list):
        raise TypeError("uniq_ids must be a list of strings.")

    # Return an empty DataFrame if uniq_ids is empty
    if not uniq_ids:
        return pd.DataFrame(columns=columns if columns else [])

    # Default columns if none are provided
    if columns is None:
        columns = ['product_name', 'brand', 'retail_price', 'discounted_price', 
                   'discount', 'description', 'product_specifications', 'overall_rating']
    
    # Ensure columns is a list of strings
    if not all(isinstance(col, str) for col in columns):
        raise ValueError("columns must be a list of strings.")

    # Initialize Supabase connection
    supabase = initialising_supabase()

    # Fetch full product details based on the uniq_ids from Supabase
    product_data = (
        supabase
        .table('flipkart_cleaned')
        .select(', '.join(columns))
        .in_('uniq_id', uniq_ids)
        .execute()
    )

    # Convert the results into a DataFrame
    product_df = pd.DataFrame(product_data.data)
    
    return product_df

''' content based '''
def fetch_content_recommendation(extracted_info, brand_preference=None, specs_preference=None):
    print("Inside content recommender")

    supabase = initialising_supabase() 

    user_query = get_user_query(extracted_info)

    filtered_products = fetch_filtered_products(extracted_info)
    content_recommendations = recommend_top_products(user_query, filtered_products)
    
    # print("content rec ends")
    # print(type(content_recommendations))
    
    # Apply brand and specification filters
    '''
    if brand_preference:
        content_recommendations = [
            rec for rec in content_recommendations 
            if brand_preference.lower() in rec['product_name'].lower()
        ]
    if specs_preference:
        content_recommendations = [
            rec for rec in content_recommendations 
            if specs_preference.lower() in rec['description'].lower()
        ]
    '''
    return content_recommendations

''' collaborative '''
def fetch_collaborative_recommendation(user_id, extracted_info, brand_preference=None, specs_preference=None):
    print("Inside collaborative recommender")

    filtered_products = fetch_filtered_products(extracted_info)

    filtered_products_df = pd.DataFrame(filtered_products)
    
    collaborative_recommendations = svd_recommend_surprise(user_id, filtered_products_df)

    '''
    # Apply brand and specification filters
    if brand_preference:
        collaborative_recommendations = collaborative_recommendations[
            collaborative_recommendations['product_name'].str.contains(brand_preference, case=False)
        ]
    if specs_preference:
        collaborative_recommendations = collaborative_recommendations[
            collaborative_recommendations['description'].str.contains(specs_preference, case=False)
        ]
    '''
    return collaborative_recommendations

''' hybridisation helper functions '''
def normalize_collaborative_scores(collaborative_recommendations):
    if not collaborative_recommendations.empty and 'predicted_rating' in collaborative_recommendations.columns:
        scaler = MinMaxScaler()
        collaborative_recommendations['normalized_predicted_rating'] = scaler.fit_transform(
            collaborative_recommendations[['predicted_rating']]
        )
    else:
        print("Warning: No valid predicted_rating found in collaborative filtering data.")
        collaborative_recommendations['normalized_predicted_rating'] = 0  # Fallback
    return collaborative_recommendations

def calculate_final_scores(content_recommendations, collaborative_recommendations, content_weight=20, collaborative_weight=0.3, top_n = 10):
    # Weight scores and perform outer merge
    content_recommendations['weighted_similarity_score'] = (content_recommendations['similarity_score'] + 1) * content_weight

    collaborative_recommendations['weighted_predicted_rating'] = collaborative_recommendations['normalized_predicted_rating'] * collaborative_weight
    print("line 245: ", collaborative_recommendations)
    hybrid = pd.merge(content_recommendations, collaborative_recommendations, on='uniq_id', how='outer')

    hybrid = hybrid.fillna(0)

    print("line 247: ", hybrid)
    hybrid['final_score'] = hybrid['weighted_similarity_score'].fillna(0) + hybrid['weighted_predicted_rating'].fillna(0)

    return hybrid.nlargest(top_n, 'final_score')

''' hybrid recommendation system '''
def hybrid_recommendations(extracted_info, user_id, content_weight=20, collaborative_weight=0.5, brand_preference=None, specs_preference=None, top_n=10):
    # Step 1: Fetch filtered products and order data
    filtered_products = fetch_filtered_products(extracted_info)
    orderdata = load_order_data()

    # Step 2: Fetch content and collaborative recommendations concurrently
    with ThreadPoolExecutor() as executor:
        content_future = executor.submit(
            fetch_content_recommendation, extracted_info, brand_preference, specs_preference
        )
        collaborative_future = executor.submit(
            fetch_collaborative_recommendation, user_id, extracted_info, brand_preference, specs_preference
        )

        try:
            content_recommendations = content_future.result()
            print("line 267: ", content_recommendations)
        except Exception as e:
            print(f"Error in content recommendations: {e}")
            content_recommendations = []  # Default or empty value if there's an error

        try:
            collaborative_recommendations = collaborative_future.result()
            print("line 269: ", collaborative_recommendations)
        except Exception as e:
            print(f"Error in collaborative recommendations: {e}")
            collaborative_recommendations = []  # Default or empty value if there's an error

    # Step 3: Normalize collaborative scores if available
    if not collaborative_recommendations.empty:
        collaborative_recommendations = normalize_collaborative_scores(collaborative_recommendations)
    print("collaborative recs", collaborative_recommendations)

    # Step 4: Calculate final scores and get top recommendations
    top_n_recommendations = calculate_final_scores(
        content_recommendations, collaborative_recommendations, content_weight, collaborative_weight, top_n
    )
    print("line 284, top_n_recommendations: ", top_n_recommendations)

    # Step 5: Fetch product details based on top recommendations
    '''
    uniq_ids = [rec['uniq_id'] for rec in top_n_recommendations]  # If structured as a list of dicts
    '''
    
    uniq_ids = top_n_recommendations['uniq_id'].tolist()  # Extract 'uniq_id' as a list from DataFrame

    product_details_df = get_product_details_from_supabase(uniq_ids)
    print(" weighted line 286: ", product_details_df)

    print("hybrid rec sys items", product_details_df['product_name'] if not product_details_df.empty else "No products found")
    return product_details_df

'''test'''
import time
extracted_info = {
    "Product Item": "skirt",
    "Budget": "2000",
    "Brand": "",
    "Product Details": "No preference"
}

user_id = "U00964"
content_weight = 0.9
collaborative_weight = 0

# Start timer
start_time = time.time()

recs = hybrid_recommendations(extracted_info, user_id, content_weight, collaborative_weight, brand_preference=None, specs_preference=None)

# End timer
end_time = time.time()
# Calculate and print the elapsed time
elapsed_time = end_time - start_time
print(f"Execution time: {elapsed_time:.2f} seconds")














''' everything in one function '''
'''
# combination
def hybrid_recommendations(extracted_info, table_name, user_id, content_weight, collaborative_weight, brand_preference = None, specs_preference = None, n = 5):
    supabase = initialising_supabase()

    filtered_products = filter_products(table_name, **extracted_info)
    orderdata = pd.DataFrame(supabase.table('synthetic_v2').select('*').execute().data)

    def fetch_content_recommendation():
        print("inside content recommender")

        keys = ['product_name', 'brand', 'product_specifications']
        user_query = concatenate_selected_dict_values(extracted_info, keys)
        content_recommendations = recommend_top_products(user_query, filtered_products)

        # Apply brand and specification filters to content-based recommendations
        if brand_preference:
            content_recommendations = [rec for rec in content_recommendations 
                                       if brand_preference.lower() in rec['product_name'].lower()]
        if specs_preference:
            content_recommendations = [rec for rec in content_recommendations 
                                       if specs_preference.lower() in rec['description'].lower()]

        return content_recommendations

    def fetch_collaborative_recommendation():
        print("inside collaborative recommender")

        collaborative_recommendations = svd_recommend_surprise_filtered(user_id, extracted_info, n_recommendations=20)

        # Apply brand and specification filters to collaborative recommendations
        if brand_preference:
            collaborative_recommendations = collaborative_recommendations[
                collaborative_recommendations['product_name'].str.contains(brand_preference, case=False)
            ]
        if specs_preference:
            collaborative_recommendations = collaborative_recommendations[
                collaborative_recommendations['description'].str.contains(specs_preference, case=False)
            ]

        return collaborative_recommendations
  
    with concurrent.futures.ThreadPoolExecutor() as executor:
        content_recommendations_future = executor.submit(fetch_content_recommendation)
        collaborative_recommendations_future = executor.submit(fetch_collaborative_recommendation)

        content_recommendations = content_recommendations_future.result()
        collaborative_recommendations = collaborative_recommendations_future.result()

    content_recommendations = fetch_content_recommendation()

    collaborative_recommendations = fetch_collaborative_recommendation()

    if not collaborative_recommendations.empty and 'predicted_rating' in collaborative_recommendations.columns and collaborative_recommendations['predicted_rating'].notna().any():
        # Normalize the predicted ratings to a scale of 0-1
        scaler = MinMaxScaler()
        collaborative_recommendations['normalized_predicted_rating'] = scaler.fit_transform(collaborative_recommendations[['predicted_rating']])
    else:
        print("Warning: No valid predicted_rating found in collaborative filtering data.")
        collaborative_recommendations['normalized_predicted_rating'] = 0  # Fallback 

    # Multiply the similarity score by content_weight and the normalized predicted rating by collaborative_weight
    content_recommendations['weighted_similarity_score'] = content_recommendations['similarity_score'] * content_weight

    collaborative_recommendations['weighted_predicted_rating'] = collaborative_recommendations['normalized_predicted_rating'] * collaborative_weight

    # Perform an outer merge to include all products from both dataframes
    hybrid = pd.merge(content_recommendations, collaborative_recommendations, left_on='product_id', right_on='uniq_id', how='outer')

    # Combine the weighted scores for both content-based and collaborative-based recommendations
    hybrid['final_score'] = hybrid['weighted_similarity_score'].fillna(0) + hybrid['weighted_predicted_rating'].fillna(0)

    # Rank the products based on the final_score
    top_n_recommendations = hybrid.nlargest(n, 'final_score')

    # Get the uniq_ids from the top recommendations
    uniq_ids = top_n_recommendations['product_id'].tolist()

    # Fetch full product details for these uniq_ids from Supabase
    product_details_df = get_product_details_from_supabase(uniq_ids)
    print("end of hybrid recommendation system")
    return product_details_df
'''


''' previous iteration - LSA_matrix '''
'''
def hybrid_recommendations(item, user_id, orderdata, lsa_matrix, content_weight, collaborative_weight, 
                           brand_preference=None, specs_preference=None, n_recommendations=10):
    def fetch_content_recommendation():
        content_recommendations = get_recommendations(item, lsa_matrix)

        # Apply brand and specification filters to content-based recommendations
        if brand_preference:
            content_recommendations = [rec for rec in content_recommendations 
                                       if brand_preference.lower() in rec['product_name'].lower()]
        if specs_preference:
            content_recommendations = [rec for rec in content_recommendations 
                                       if specs_preference.lower() in rec['description'].lower()]

        return content_recommendations
    
    def fetch_collaborative_recommendation():
        collaborative_recommendations = svd_recommend_surprise(user_id, n_recommendations)

        # Apply brand and specification filters to collaborative recommendations
        if brand_preference:
            collaborative_recommendations = collaborative_recommendations[
                collaborative_recommendations['product_name'].str.contains(brand_preference, case=False)
            ]
        if specs_preference:
            collaborative_recommendations = collaborative_recommendations[
                collaborative_recommendations['description'].str.contains(specs_preference, case=False)
            ]

        return collaborative_recommendations
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        content_recommendations_future = executor.submit(fetch_content_recommendation)
        collaborative_recommendations_future = executor.submit(fetch_collaborative_recommendation)

        content_recommendations = content_recommendations_future.result()
        collaborative_recommendations = collaborative_recommendations_future.result()

    product_info = {}
    for item in content_recommendations:
        product_name = item['product_name']
        score = float(item['similarity_score']) * content_weight
        price = item['retail_price']
        description = item['description']
        product_info[product_name] = [score, price, description]
    min_rating = 0
    max_rating = 5

    for index, row in collaborative_recommendations.iterrows():
        product_name = row['product_name']
        raw_score = float(row['predicted_rating'])
        normalized_score = (raw_score - min_rating) / (max_rating - min_rating)
        score = normalized_score * collaborative_weight
        price = row['retail_price']
        description = row['description']
        if product_name in product_info:
            product_info[product_name][0] += score
        else:
            product_info[product_name] = [score,  price, description]

    for product_name, info in product_info.items():
        info[0] = (info[0] * max_rating) - min_rating
    product_list = [(product_name, info) for product_name, info in product_info.items()]
    sorted_product_list = sorted(product_list, key=lambda x: x[1][0], reverse=True)
    sorted_product_info = {product_name: info for product_name, info in sorted_product_list}

    top_recommendations = list(sorted_product_info.items())[:n_recommendations]
    final_output = []
    for product_name, info in top_recommendations:
        score = info[0] 
        price = info[1] 
        description = info[2]
    
        final_output.append({
            'product_name': product_name,
            'price': price,
            'description': description,
            'predicted_rating': score  
        })
    return final_output

'''

'''
def hybrid_recommendations(catalogue, item, user_id, orderdata, lsa_matrix, content_weight, collaborative_weight, 
                           brand_preference=None, specs_preference=None, n_recommendations=10):
    #lsa_matrix = get_lsa_matrix(catalogue, lsa_matrix_file)

    def fetch_content_recommendation():
        content_recommendations = get_recommendations(item, catalogue, lsa_matrix)

        # Apply brand and specification filters to content-based recommendations
        if brand_preference:
            content_recommendations = [rec for rec in content_recommendations 
                                       if brand_preference.lower() in rec['product_name'].lower()]
        if specs_preference:
            content_recommendations = [rec for rec in content_recommendations 
                                       if specs_preference.lower() in rec['description'].lower()]

        return content_recommendations
    
    def fetch_collaborative_recommendation():
        collaborative_recommendations = svd_recommend_surprise(catalogue, user_id, orderdata, n_recommendations)

        # Apply brand and specification filters to collaborative recommendations
        if brand_preference:
            collaborative_recommendations = collaborative_recommendations[
                collaborative_recommendations['product_name'].str.contains(brand_preference, case=False)
            ]
        if specs_preference:
            collaborative_recommendations = collaborative_recommendations[
                collaborative_recommendations['description'].str.contains(specs_preference, case=False)
            ]

        return collaborative_recommendations
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        content_recommendations_future = executor.submit(fetch_content_recommendation)
        collaborative_recommendations_future = executor.submit(fetch_collaborative_recommendation)

        content_recommendations = content_recommendations_future.result()
        collaborative_recommendations = collaborative_recommendations_future.result()

    product_info = {}
    for item in content_recommendations:
        product_name = item['product_name']
        score = float(item['similarity_score']) * content_weight
        price = item['retail_price']
        description = item['description']
        product_info[product_name] = [score, price, description]
    min_rating = 0
    max_rating = 5

    for index, row in collaborative_recommendations.iterrows():
        product_name = row['product_name']
        raw_score = float(row['predicted_rating'])
        normalized_score = (raw_score - min_rating) / (max_rating - min_rating)
        score = normalized_score * collaborative_weight
        price = row['retail_price']
        description = row['description']
        if product_name in product_info:
            product_info[product_name][0] += score
        else:
            product_info[product_name] = [score,  price, description]

    for product_name, info in product_info.items():
        info[0] = (info[0] * max_rating) - min_rating
    product_list = [(product_name, info) for product_name, info in product_info.items()]
    sorted_product_list = sorted(product_list, key=lambda x: x[1][0], reverse=True)
    sorted_product_info = {product_name: info for product_name, info in sorted_product_list}

    top_recommendations = list(sorted_product_info.items())[:n_recommendations]
    final_output = []
    for product_name, info in top_recommendations:
        score = info[0] 
        price = info[1] 
        description = info[2]
    
        final_output.append({
            'product_name': product_name,
            'price': price,
            'description': description,
            'predicted_rating': score  
        })
    return final_output
'''
"""
user_id = 'U06610'
user_product = 'pants'
n_recommendations = 20
content_weight = 0.3
collaborative_weight = 0.7

recommendations = hybrid_recommendations(user_product, user_id, df, lsa_matrix, orderdata, content_weight, collaborative_weight, n_recommendations=10)

for recommendation in recommendations:
    print(f"Product Name: {recommendation['product_name']}, Predicted Rating: {recommendation['predicted_rating']}")

"""

# content_weights = np.linspace(0, 1, 11)  # [0, 0.1, 0.2, ..., 1]
# collaborative_weights = np.linspace(0, 1, 11)

# best_rmse = float('inf')
# best_weights = (0, 0)

# def evaluate_hybrid_system(user_product, user_id, df, lsa_matrix, orderdata, true_ratings, n_recommendations, content_weight, collaborative_weight):
#     recommendations = hybrid_recommendations(user_product, user_id, df, lsa_matrix, orderdata, content_weight, collaborative_weight, n_recommendations=10)

#     predicted_ratings = []
#     for rec in recommendations:
#         predicted_ratings.append(rec['score'])
#     rmse = np.sqrt(mean_squared_error(true_ratings, predicted_ratings))
#     return rmse

# for cw in content_weights:
#     for cf in collaborative_weights:
#         if cw + cf == 1:  # Ensure the weights sum to 1
#             rmse = evaluate_hybrid_system(user_product, user_id, df, lsa_matrix, orderdata, true_ratings, n_recommendations, cw, cf)
#             if rmse < best_rmse:
#                 best_rmse = rmse
#                 best_weights = (cw, cf)

# print(f"Best weights: Content Weight: {best_weights[0]}, Collaborative Weight: {best_weights[1]}")
# print(f"Best RMSE: {best_rmse}")


