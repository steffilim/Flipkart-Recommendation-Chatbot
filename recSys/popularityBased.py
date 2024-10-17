import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

### Targeted at new customers, built on demographic filtering to resolve the cold start problem

# Loading the dataset
df = pd.read_csv('newData/synthetic_v2.csv')
df['Order Date'] = pd.to_datetime(df['Order Date'])

catalog = pd.read_csv('newData/flipkart_cleaned.csv')
catalog.rename(columns={'uniq_id': 'Product ID'}, inplace=True)


# Get purchases within the past year
one_year_ago = datetime.now() - timedelta(days=365)
recent_purchases = df[df['Order Date'] >= one_year_ago]

# Getting trending products (by sale volume)
popular_products = pd.DataFrame(recent_purchases.groupby('Product ID')['User rating for the product'].count())
top_5_most_popular = popular_products.sort_values('User rating for the product', ascending=False).head()

# finding the product name and saving it into the top_5_most_popular dataframe
detailed_top_products = pd.merge(top_5_most_popular, catalog, on='Product ID', how='left')
detailed_top_products = detailed_top_products[['product_name', 'discounted_price', 'description', 'User rating for the product']]

print(detailed_top_products.columns)

# saving to dataframe for easy retrieval
detailed_top_products.to_csv('newData/top_5_most_popular.csv', index=False)