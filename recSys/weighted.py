from collaborative import svd_recommend_surprise, svd_recommend_surprise_filtered
from contentBased import recommend_top_products
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
from dotenv import load_dotenv
import os
import pymongo
import concurrent.futures
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
def initialising_supabase():
    load_dotenv()
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_API_KEY = os.getenv("SUPABASE_API_KEY")
    supabase = create_client(SUPABASE_URL, SUPABASE_API_KEY)
    return supabase

def filter_products(table_name, product_name=None, price_limit=None, brand=None, overall_rating=None, product_specifications=None):
    # Build the SQL query dynamically based on the filters provided
    supabase = initialising_supabase()
    query = supabase.table(table_name).select("*")
    
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

def concatenate_selected_dict_values(input_dict, keys_to_extract, separator=", "):
    result = ''
    
    # Iterate over the list of keys to extract values
    for key in keys_to_extract:
        if key in input_dict:
            result += str(input_dict[key]) + separator
    
    # Remove the trailing separator
    result = result.rstrip(separator)
    
    return result

def get_product_details_from_supabase(uniq_ids):
    # Initialize Supabase connection
    supabase = initialising_supabase()
    
    # Fetch full product details based on the uniq_ids from Supabase
    product_data = supabase.table('flipkart_cleaned').select('*').in_('uniq_id', uniq_ids).execute()

    # Convert the results into a DataFrame
    product_df = pd.DataFrame(product_data.data)
    
    return product_df

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



'''test'''
extracted_info = extracted_info = {
    "product_name": "cycling shorts",
    "price_limit": 3000,
    "brand": "alisha",
    "overall_rating": '0',
    "product_specifications": ''
}
table_name = 'flipkart_cleaned'
user_id = 'U01357'
content_weight = 0.6
collaborative_weight = 0.4

# Start timer
start_time = time.time()

prods = hybrid_recommendations(extracted_info, table_name, user_id, content_weight, collaborative_weight, brand_preference = None, specs_preference = None)
print("products", prods)


# End timer
end_time = time.time()
# Calculate and print the elapsed time
elapsed_time = end_time - start_time
print(f"Execution time: {elapsed_time:.2f} seconds")

#re ranking with intent



''' previous iteration '''
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


