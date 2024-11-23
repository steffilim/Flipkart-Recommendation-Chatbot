import pandas as pd
from datetime import datetime, timedelta
from functions.databaseFunctions import *

def get_trending_products():
    """
    Identifies trending products from recent purchase data based on sales volume.

    Returns:
        pd.DataFrame: A DataFrame containing the top 5 most popular products, 
                      including their details (name, price, description, and ratings).

    Notes:
        The time frame for "trending" is set to one year from the current date.
    """

    # Fetch the catalogue & order data from Supabase
    catalogue = load_product_data()
    orderdata = load_order_data()  

    # Filter for purchases within the past year
    one_year_ago = datetime.now() - timedelta(days=365)
    recent_purchases = orderdata[pd.to_datetime(orderdata['Order Date']) >= one_year_ago]

    # Identify trending products by sale volume
    popular_products = recent_purchases.groupby('uniq_id')['User rating for the product'].count().reset_index()
    top_5_most_popular = popular_products.sort_values('User rating for the product', ascending=False).head()

    # Add product details from the catalog
    detailed_top_products = pd.merge(top_5_most_popular, catalogue, on='uniq_id', how='left')
    detailed_top_products = detailed_top_products[['uniq_id', 'product_name', 'discounted_price', 'description', 'User rating for the product']]
    
    return detailed_top_products