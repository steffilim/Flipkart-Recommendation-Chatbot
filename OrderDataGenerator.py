import pandas as pd
import numpy as np
import random
from scipy.stats import truncnorm

# Load your dataset (replace 'your_dataset.csv' with the actual dataset file path)
df = pd.read_csv('newData/flipkart_cleaned.csv')

# Adjusted parameters for the dataset
num_users = 10000  # You can change this number to increase or decrease users
num_orders = 50000  # You can change this number to increase or decrease orders

def generate_truncated_normal(mean, std_dev, lower_bound, upper_bound, size):
    lower_limit = (lower_bound - mean) / std_dev
    upper_limit = (upper_bound - mean) / std_dev
    return truncnorm.rvs(lower_limit, upper_limit, loc=mean, scale=std_dev, size=size).astype(int)


age_mean = 40  # Average age of users
age_std_dev = 12  # Standard deviation for age

income_mean = 80000  # Average income of users
income_std_dev = 30000  # Standard deviation for income

user_ages = generate_truncated_normal(age_mean, age_std_dev, 18, 70, num_users)
user_incomes = generate_truncated_normal(income_mean, income_std_dev, 30000, 200000, num_users)

# Generate user IDs
user_ids = [f'U{str(i).zfill(5)}' for i in range(1, num_users + 1)]

# Generate synthetic user data
occupations = ["Engineer", "Doctor", "Artist", "Teacher", "Student", "Lawyer", "Accountant", "Scientist", "Manager"]
user_occupations = np.random.choice(occupations, size=num_users)
interests = ["Electronics", "Fashion", "Books", "Sports", "Music", "Travel", "Cooking", "Fitness"]
user_interests = [random.sample(interests, k=random.randint(1, 3)) for _ in range(num_users)]
ethnicities = ["Asian", "Caucasian", "Hispanic", "African American", "Other"]
user_ethnicities = np.random.choice(ethnicities, size=num_users)
passwords = [''.join(random.choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()', k=12)) for _ in range(num_users)]

# Create a DataFrame for users
users_df = pd.DataFrame({
    "User ID": user_ids,
    "User Age": user_ages,
    "User Occupation": user_occupations,
    "User Income": user_incomes,
    "User Interests": ["; ".join(interest) for interest in user_interests],
    "User Ethnicity": user_ethnicities
})

# Initialize list for orders and previous ratings tracker
orders = []
previous_ratings = {user_id: {} for user_id in user_ids}


# Function to generate user ratings with patterns
def generate_user_rating(user_id, product, previous_ratings, user_bias=None):
    brand = product['brand']
    base_rating = np.random.normal(3, 1)  # Start with a normal distribution around 3
    if brand in previous_ratings[user_id]:
        # Favor brands the user has rated before
        base_rating = np.mean(previous_ratings[user_id][brand]) + np.random.uniform(-0.5, 0.5)
    
    overall_rating = product['overall_rating']
    if overall_rating != 'No rating available':
        try:
            overall_rating_value = float(overall_rating)
            # Weigh user's base rating with the product's overall rating
            base_rating = (0.7 * base_rating + 0.3 * overall_rating_value)
        except ValueError:
            pass
    
    # Add some user-specific bias (e.g., lenient or harsh rater)
    if user_bias:
        base_rating += user_bias

    # Bound the rating between 0 and 5
    rating = min(max(base_rating, 0), 5)
    return round(rating, 1)

# Example: Assigning user bias to favor high or low ratings
user_biases = np.random.normal(0, 0.5, size=num_users)  # Some users are stricter (-ve bias), some lenient (+ve bias)



order_id_counter = 1

# Generate orders
for _ in range(num_orders):
    user_id = random.choice(user_ids)
    num_products = np.random.randint(1, 5)
    products = df.sample(num_products)

    order_total = (products['discounted_price'] * np.random.randint(1, 4, size=num_products)).sum()
    order_date = pd.to_datetime('2023-01-01') + pd.to_timedelta(np.random.randint(1, 365), unit='d')
    purchase_address = f"{random.randint(100, 999)} {random.choice(['Elm St.', 'Main St.', 'Oak St.', 'Pine St.'])}{random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston'])}"

    for _, product in products.iterrows():
        rating = generate_user_rating(user_id, product, previous_ratings)

        orders.append({
            "Order ID": f'O{str(order_id_counter).zfill(7)}',
            "Product ID": product['uniq_id'],
            "Product Quantity": np.random.randint(1, 4),
            "Product Price Each": product['discounted_price'],
            "Order Total": order_total,
            "Order Date": order_date.strftime('%Y-%m-%d'),
            "Purchase Address": purchase_address,
            "User rating for the product": rating,
            "User ID": user_id,
            "User Age": users_df[users_df['User ID'] == user_id]['User Age'].values[0],
            "User Occupation": users_df[users_df['User ID'] == user_id]['User Occupation'].values[0],
            "User Income": users_df[users_df['User ID'] == user_id]['User Income'].values[0],
            "User Interests": users_df[users_df['User ID'] == user_id]['User Interests'].values[0],
            "User Ethnicity": users_df[users_df['User ID'] == user_id]['User Ethnicity'].values[0]
        })
    order_id_counter += 1

# Convert to DataFrame
orders_df = pd.DataFrame(orders)

# Save the dataset as CSV
orders_df.to_csv('newData/synthetic_v2.csv', index=False)

print("Dataset generated and saved as 'synthetic_v2.csv'")
