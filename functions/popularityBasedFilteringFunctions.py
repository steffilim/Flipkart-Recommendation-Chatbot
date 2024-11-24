from functions.databaseFunctions import *

def get_most_recent_purchase(user_id):
    """
    Retrieves the most recent purchase for a given user.

    Args:
        user_id (str): The ID of the user.

    Returns:
        tuple: A tuple containing the unique ID of the most recent purchase and the order date. 
               Returns (None, None) if no purchase history exists.
    """

    supabase = initialising_supabase()
    users_data = load_order_data(supabase)
    users_data = users_data[users_data['User ID'] == user_id].sort_values(by='Order Date', ascending=False)
    if not users_data.empty:
        most_recent_uniq_id = users_data.iloc[0]['uniq_id']
    else:
        return None, None
    return most_recent_uniq_id, users_data.iloc[0]['Order Date']

def get_similar_products(uniq_id):
    """
    Finds similar products based on category and overall rating.

    Args:
        uniq_id (str): The unique ID of the product.

    Returns:
        pd.DataFrame or list: A DataFrame of the top 3 similar products, or an empty list if no similar products are found.
    """

    supabase = initialising_supabase()
    catalogue_data = pd.DataFrame(supabase.table('flipkart_cleaned_2k').select('*').execute().data)
    catalogue_data['content'] = catalogue_data['description'].astype(str) + ' ' + catalogue_data['product_specifications'].astype(str)
    catalogue_data['content'] = catalogue_data['content']. fillna("")
    catalogue_data['overall_rating'] = pd.to_numeric(catalogue_data['overall_rating'], errors='coerce')
    catalogue_data['overall_rating'].fillna(2, inplace=True)
    product_row = catalogue_data[catalogue_data['uniq_id'] == uniq_id]
    
    if product_row.empty:
        return []
    product_category_tree = ast.literal_eval(product_row.iloc[0]['product_category_tree'])[0]

    similar_products = catalogue_data[
        (catalogue_data['product_category_tree'].apply(lambda x: ast.literal_eval(x)[0] if x else "") == product_category_tree) &
        (catalogue_data['uniq_id'] != uniq_id)
    ].nlargest(3, 'overall_rating')

    return similar_products

def recommend_similar_products(user_id):
    """
    Recommends products similar to the user's most recent purchase.

    Args:
        user_id (str): The ID of the user.

    Returns:
        str or list[str]: A message with recommendations or a default prompt if no history exists.
    """

    most_recent_uniq_id, order_date = get_most_recent_purchase(user_id)
    if most_recent_uniq_id is None:
        return "No purchase history found for this user."
    similar_products = get_similar_products(most_recent_uniq_id)
    
    if isinstance(similar_products, pd.DataFrame) and not similar_products.empty:
        recommendations = []
        recommendations = [
            f"{idx + 1}. {row['product_name']} at ₹{row['discounted_price']}\n\n"
            f"Description: {row['description']}\n"
            for idx, (_, row) in enumerate(similar_products.iterrows())
        ]

        # Combine recommendations into a single response text
        response_text = recommendations
        return response_text
    else:
        return "Welcome back! What are you looking for today?"


def popular_items():
    """
    Retrieves the top 5 popular products from the database.

    Returns:
        list[str]: A list of strings containing product details and their descriptions.
    """

    supabase = initialising_supabase()
    top5 = pd.DataFrame(supabase.table('top5products').select('*').execute().data) 

    popular_items = []

    for index, product in top5.iterrows():
        item_details = f"{index + 1}. {product['product_name']} at {INR}{product['discounted_price']} \n\n Description: {product.get('description', 'No description available')} \n\n"
        popular_items.append(item_details)
    return popular_items

def get_popular_items():
    """
    Generates a response with the week's popular items and prompts the user for further interaction.

    Returns:
        str: A response text listing popular items.
    """

    popularitems = popular_items()
    response_text = "Here are these week's popular items:\n" + "\n".join(popularitems)
    response_text += "\n\nWould you like to know more about any of these items? If not, please provide me the description of the item you are looking for."

    return response_text

def getting_user_purchase_dictionary(user_id, supabase):
    """
    Retrieves the user's profile and purchase history from the database.

    Args:
        user_id (str): The ID of the user.
        supabase (supabase.Client): An initialized Supabase client.

    Returns:
        tuple: A tuple containing the user profile and a list of past purchases. 
               Returns a message if no purchase history is found.
    """

    user_profile = supabase.table('synthetic_v2_2k').select('User Age', 'User Occupation', 'User Interests').eq('User ID', user_id).execute().data

    user_purchase_data = supabase.table('synthetic_v2_2k').select('uniq_id', 'User rating for the product') \
                        .eq('User ID', user_id).execute().data
    
    
    # Check if purchase history exists
    if not user_purchase_data:
        return f"No purchase history found for user ID {user_id}."

    # Extract product IDs from purchase history
    product_ids = [purchase['uniq_id'] for purchase in user_purchase_data]

    # Query Supabase for product details based on the extracted product IDs
    product_details = supabase.table('flipkart_cleaned_2k') \
                        .select('product_name, uniq_id') \
                        .in_('uniq_id', product_ids) \
                        .execute().data

    purchase_history_merge = pd.merge(pd.DataFrame(user_purchase_data), pd.DataFrame(product_details), how='inner', on='uniq_id')
    
    # Process each purchase and its product details
    user_purchases = []
    for items in purchase_history_merge.iterrows():
        user_purchases.append((items[1]['product_name'], items[1]['User rating for the product']))
        
    return user_profile, user_purchases