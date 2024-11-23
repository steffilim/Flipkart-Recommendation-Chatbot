import numpy as np
from joblib import dump, load
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
from functions.databaseFunctions import *

# Load the Sentence Transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
# model.save("sentenceTransformer")

''' computing embeddings '''
def precompute_product_embeddings(df, batch_size=1000, print_interval=100):
    """
    Precomputes embeddings for product descriptions using a SentenceTransformer model.

    Args:
        df (pd.DataFrame): DataFrame containing product information with columns:
            - 'content' (str): Text description of the product.
            - 'uniq_id' (str): Unique identifier for each product.
        batch_size (int, optional): Number of products to process in each batch. Default is 1000.
        print_interval (int, optional): Interval at which progress updates are printed. Default is 100.

    Returns:
        list[dict]: List of dictionaries containing product IDs and their corresponding embeddings.
    """

    product_descriptions = df['content'].tolist()
    product_ids = df['uniq_id'].tolist()
    total_products = len(product_descriptions)
    
    product_embeddings = []
    
    # Process embeddings in batches
    for start in range(0, total_products, batch_size):
        end = min(start + batch_size, total_products)
        batch_descriptions = product_descriptions[start:end]
        batch_ids = product_ids[start:end]
        
        # Compute embeddings for the batch
        with torch.no_grad():  # Avoid computing gradients to save memory
            batch_embeddings = model.encode(batch_descriptions, convert_to_tensor=True)
        
        # Store embeddings with their associated product IDs
        product_embeddings.extend(
            {"product_id": pid, "embedding": embedding} for pid, embedding in zip(batch_ids, batch_embeddings)
        )
        
        # Print progress for every print_interval items processed
        if (end % print_interval == 0) or (end == total_products):
            print(f"Processed {end}/{total_products} products...")

        print("precompute_product_embeddings end")

    return product_embeddings

''' storing data - into supabase'''
def store_product_embeddings_in_supabase(product_embeddings):
    """
    Stores precomputed product embeddings into the Supabase database.

    Args:
        product_embeddings (list[dict]): List of dictionaries containing:
            - 'product_id' (str): Unique identifier for the product.
            - 'embedding' (torch.Tensor or list): Embedding for the product.

    Returns:
        None
    """

    supabase = initialising_supabase()

    for entry in product_embeddings:
        pid = entry['product_id']
        embedding = entry['embedding']

        # Convert embedding tensor to a list of floats if it's a tensor
        embedding_list = embedding.cpu().numpy().tolist() if isinstance(embedding, torch.Tensor) else embedding
        
        # Convert embedding list to a string (comma-separated values)
        embedding_str = ','.join(map(str, embedding_list))

        # Insert product_id and corresponding embedding string into Supabase
        response = supabase.table("embeddings").upsert({
            "product_id": pid,
            "embedding": embedding_str  # Storing the embedding as a string
        }).execute()

        # Debug: Print the response from Supabase
        print(f"Response from Supabase: {response}")
    
    print("Finished storing product embeddings in Supabase.")

''' retrieving embeddings for a list of product_ids'''
def get_product_embeddings(product_ids):
    """
    Fetches embeddings for the given product IDs from the Supabase database.

    Args:
        product_ids (list[str]): List of product IDs for which embeddings are to be fetched.
                                 Product IDs cannot be empty strings.

    Returns:
        pd.DataFrame: DataFrame containing columns:
            - 'product_id' (str): Unique identifier for the product.
            - 'embedding_list' (list[float]): List of floats representing the embedding.
    """
    supabase = initialising_supabase()
    
    # Input validation
    if not isinstance(product_ids, list):
        raise TypeError("product_ids must be a list")

    # Return empty DataFrame immediately if empty list
    if not product_ids:
        print("Empty input list provided")
        return pd.DataFrame(columns=["product_id", "embedding_list"])
    
    # Validate all items are strings
    if not all(isinstance(pid, str) for pid in product_ids):
        raise ValueError("All product IDs must be strings")
    
    # Initialize response outside the loop
    response = None
    response = supabase.table("product_embeddings") \
                      .select("product_id, embedding_list") \
                      .in_("product_id", product_ids) \
                      .execute()
        
    df = pd.DataFrame(columns=["product_id", "embedding_list"])

    if response and response.data:
        df = pd.DataFrame(response.data)
    else:
        print('no matching data found in product embeddings')
    
    return df

''' convertion of embeddings list 'text' to 'float' (cannot save as array of 32 bits in supabase)'''
# remove brackets from embedding list
def remove_brackets_from_embedding_list(df):
    """
    Removes brackets from the 'embedding_list' column in a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing a column 'embedding_list' with string representations of lists.

    Returns:
        pd.DataFrame: DataFrame with the 'embedding_list' column updated to remove brackets.
    """

    if 'embedding_list' in df.columns:        
        # Remove brackets from each entry in embedding_list
        df['embedding_list'] = df['embedding_list'].str.replace('[', '', regex=False).str.replace(']', '', regex=False)
    
    return df

# convertion from list to float
def convert_embedding_list_to_floats(df):
    """
    Converts a comma-separated string of numbers in the 'embedding_list' column 
    to a list of floats.

    Args:
        df (pd.DataFrame): DataFrame containing a column 'embedding_list' with string representations of lists.

    Returns:
        pd.DataFrame: DataFrame with the 'embedding_list' column updated to contain lists of floats.
    """
    if 'embedding_list' in df.columns:        
        # Convert the comma-separated string to a list of floats
        df['embedding_list'] = df['embedding_list'].apply(
            lambda x: list(map(float, x.split(',')))
        )
    
    return df

def check_if_embedding_list_is_float(df):
    """
    Checks if each entry in the 'embedding_list' column is a list of floats and 
    adds a new column 'is_float_list' indicating the result.

    Args:
        df (pd.DataFrame): DataFrame containing a column 'embedding_list'.

    Returns:
        pd.DataFrame: DataFrame with an additional column 'is_float_list' (bool).
    """

    # Check if embedding_list is a list of floats
    if 'embedding_list' in df.columns:
        df['is_float_list'] = df['embedding_list'].apply(
            lambda x: isinstance(x, list) and all(isinstance(i, float) for i in x)
        )
    return df

''' recommendation function '''
def recommend_top_products(user_query, filtered_products, top_n=20):
    """
    Recommends the top N products based on similarity to the user's query.

    Args:
        user_query (str): Query input by the user (e.g., "I want a red shoe"). Must be non-empty.
        filtered_products (list[dict]): List of dictionaries containing product information. 
            Each dictionary must include:
            - 'uniq_id' (str): Unique identifier for the product.
            - 'product_name' (str): Name of the product.
            - 'retail_price' (float): Price of the product.
            - 'description' (str): Description of the product.
        top_n (int, optional): Number of top recommendations to return. Default is 20.

    Returns:
        pd.DataFrame: DataFrame containing the top recommended products with columns:
            - 'uniq_id' (str): Product ID.
            - 'similarity_score' (float): Similarity score with the user's query.
    
    Raises:
        ValueError: If user_query is not a string or is empty.
        TypeError: If filtered_products is not a list of dictionaries, or if any dictionary 
                   does not contain a 'uniq_id'.
    """

    supabase = initialising_supabase()

    query_embedding = model.encode([user_query])[0]  

    # Get product IDs from filtered products
    product_ids = [product['uniq_id'] for product in filtered_products]

    # Get specific product embeddings
    raw_product_embeddings = get_product_embeddings(product_ids)

    # Converting it to float instead of string
    text_product_embeddings = remove_brackets_from_embedding_list(raw_product_embeddings)
    product_embeddings = convert_embedding_list_to_floats(text_product_embeddings)
    product_embeddings = product_embeddings['embedding_list']

    # Flatten the embedding list into a single 1D array
    product_embeddings_flat = [embedding for sublist in product_embeddings for embedding in sublist]

    # Calculate similarity scores with precomputed product embeddings
    product_embeddings_array = np.array(product_embeddings_flat)

    # Reshaping Array
    product_embeddings_array = product_embeddings_array.reshape(-1, 384)

    # Calculate similarity scores
    similarities = cosine_similarity([query_embedding], product_embeddings_array)[0]
    
    # Get top N most similar products
    sorted_indices = np.argsort(similarities)[::-1][:top_n]

    # Prepare the top products with their similarity scores
    top_products_with_scores = []
    for idx in sorted_indices:
        product = filtered_products[idx]
        similarity_score = similarities[idx]
        top_products_with_scores.append({
            'uniq_id': product['uniq_id'],
            'similarity_score': similarity_score,
        })

    # Convert to DataFrame
    df = pd.DataFrame(top_products_with_scores)

    return df

''' TEST '''
'''
# Example extracted information from user input
extracted_info = {
    "product_name": "cycling shorts",
    "max_price": 3000,
    "brand": "alisha",
    "specifications": "No preference"
}

# filtering of products

# Start the timer
start_time = time.time()

filtered_products = fetch_filtered_products(extracted_info)

top_products = recommend_top_products(extracted_info, filtered_products)
# End the timer
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - filter_time
print(f"Time taken to run the recommendation function: {elapsed_time:.4f} seconds")
top_products
'''