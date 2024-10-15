from collaborative import svd_recommend_surprise
from contentBased import get_recommendations
from contentBased import get_lsa_matrix
from contentBased import load_product_data
import pandas as pd

product_data_file = 'newData/flipkart_cleaned.csv'
lsa_matrix_file = 'lsa_matrix.joblib'

product_data = load_product_data(product_data_file)
lsa_matrix = get_lsa_matrix(product_data, lsa_matrix_file)

orderdata = pd.read_csv("newData/synthetic_v2.csv")
orderdata = orderdata.rename(columns={'Product ID': 'uniq_id'})
orderdata

def hybrid_recommendations(user_product, user_id, df, lsa_matrix, orderdata, n_recommendations=10, content_weight=0.4, collaborative_weight=0.6):
    content_recommendations = get_recommendations(user_product, df, lsa_matrix)
    
    collaborative_recommendations = svd_recommend_surprise(user_id, orderdata, n_recommendations)

    product_scores = {}

    for item in content_recommendations:
        product_name = item['product_name']
        score = item['similarity_score'] * content_weight
        product_scores[product_name] = score

    min_rating = 0
    max_rating = 5

    for index, row in collaborative_recommendations.iterrows():
        product_name = row['product_name']
        raw_score = row['predicted_rating']
        normalized_score = (raw_score - min_rating) / (max_rating - min_rating)
        score = normalized_score * collaborative_weight
        if product_name in product_scores:
            product_scores[product_name] += score
        else:
            product_scores[product_name] = score
    
    sorted_products = sorted(product_scores.items(), key=lambda x: x[1], reverse=True)

    top_recommendations = sorted_products[:n_recommendations]
    
    final_output = []
    for product_name, score in top_recommendations:
        final_output.append({
            'product_name': product_name,
            'score': score
        })
    
    return final_output

user_id = 'U06610'
user_product = 'pants'
n_recommendations = 10   
recommendations = hybrid_recommendations(user_product, user_id, product_data, lsa_matrix, orderdata, n_recommendations, content_weight=0.6, collaborative_weight=0.4)[:5]

for recommendation in recommendations:
    print(f"Product Name: {recommendation['product_name']}, Score: {recommendation['score']}")
