from recSys.collaborative import svd_recommend_surprise
from recSys.contentBased import get_recommendations
from recSys.contentBased import get_lsa_matrix
from recSys.contentBased import load_product_data
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
from dotenv import load_dotenv
import os
import pymongo
import concurrent.futures

load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI")
FLIPKART = os.getenv("FLIPKART")

client = pymongo.MongoClient(MONGODB_URI)
flipkart = client[FLIPKART]

product_data_file = flipkart.catalogue
lsa_matrix_file = 'lsa_matrix.joblib'


"""df = load_product_data(product_data_file)
lsa_matrix = get_lsa_matrix(df, lsa_matrix_file)

orderdata = pd.read_csv("newData/synthetic_v2.csv")
orderdata = orderdata.rename(columns={'Product ID': 'uniq_id'})
orderdata"""

def hybrid_recommendations(catalogue, item, user_id, orderdata,lsa_matrix, content_weight, collaborative_weight, n_recommendations=10):
    #lsa_matrix = get_lsa_matrix(catalogue, lsa_matrix_file)

    def fetch_content_recommendation():
        return get_recommendations(item, catalogue, lsa_matrix)
    
    def fetch_collaborative_recommendation():
        return svd_recommend_surprise(catalogue, user_id, orderdata, n_recommendations)
    
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


