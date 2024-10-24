from surprise import Dataset, Reader
from surprise import SVD, NMF
from surprise.model_selection import cross_validate, train_test_split, GridSearchCV
import pandas as pd
import numpy as np

"""
flipkart = pd.read_csv("newData/flipkart_cleaned.csv")
flipkart
flipkart['product_category_tree'] = flipkart['product_category_tree'].apply(lambda x: ' '.join(eval(x)).lower())

orderdata = pd.read_csv("newData/synthetic_v2.csv")
orderdata = orderdata.rename(columns={'Product ID': 'uniq_id'})
orderdata
"""

def svd_recommend_surprise(catalogue, user_id, orderdata, n_recommendations=20):
    print("SVD")
    cursor_cat = catalogue.find({})
    catalogue = pd.DataFrame(list(cursor_cat))  

    cursor_order = orderdata.find({})   
    orderdata = pd.DataFrame(list(cursor_order))
    reader = Reader(rating_scale=(0, 5))

    print(orderdata)
    print(reader)
    dataset = Dataset.load_from_df(orderdata[['User ID', 'uniq_id', 'User rating for the product']], reader)
    trainset = dataset.build_full_trainset()

    svd = SVD()
    svd.fit(trainset)

    # Filter unrated products
    user_ratings = orderdata[orderdata['User ID'] == user_id]
    user_rated_products = user_ratings['uniq_id'].values
    all_products = orderdata['uniq_id'].unique()
    unrated_products = [prod for prod in all_products if prod not in user_rated_products]

    
    predictions = []
    for product_id in unrated_products:
        pred = svd.predict(user_id, product_id)
        predictions.append((product_id, pred.est))

    sorted_predictions = sorted(predictions, key=lambda x: x[1], reverse=True)

    recommended_product_ids = [prod[0] for prod in sorted_predictions[:n_recommendations]]
    
    # Get product information and include predicted ratings
    recommendations = catalogue[catalogue['uniq_id'].isin(recommended_product_ids)].copy()
    print(recommendations)
    for idx, row in recommendations.iterrows():
        prediction = next((pred[1] for pred in predictions if pred[0] == row['uniq_id']), None)
        recommendations.loc[idx, 'predicted_rating'] = prediction
    recommendations_df = recommendations[['uniq_id', 'product_name', 'description', 'product_category_tree', 'predicted_rating', 'retail_price']]
    
    return recommendations_df

# To test
#user_id = 'U06610'
#recommendations = svd_recommend_surprise(user_id, orderdata)
#print(recommendations)
