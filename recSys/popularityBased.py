import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

### Targeted at new customers

# Loading the dataset
df = pd.read_csv('newData/synthetic_v2.csv')
df['Order Date'] = pd.to_datetime(df['Order Date'])

# Get purchases within the past year
one_year_ago = datetime.now() - timedelta(days=365)
recent_purchases = df[df['Order Date'] >= one_year_ago]

# Getting trending products (by sale volume)
popular_products = pd.DataFrame(recent_purchases.groupby('Product ID')['User rating for the product'].count())
most_popular = popular_products.sort_values('User rating for the product', ascending=False)

print(most_popular.head())