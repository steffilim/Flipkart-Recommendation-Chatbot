
""" KEYWORD DETECTION FUNCTION """

import nltk
from rake_nltk import Rake
from nltk.corpus import words
nltk.download('words')
nltk.download('stopwords')
nltk.download('punkt_tab')

# Function to check if the user's input is valid
def is_valid_input(user_input):
    word_list = set(words.words())
    tokens = nltk.word_tokenize(user_input)

    valid_tokens = [word for word in tokens if word.lower() in word_list] 
    return len(valid_tokens) > 0

def extract_keywords(item):
    r = Rake()
    r.extract_keywords_from_text(item)
    query_keyword = r.get_ranked_phrases_with_scores()
    query_keyword_ls = [keyword[1] for keyword in query_keyword]
    return query_keyword_ls


""" RECOMMENDATION FUNCTION """
# Initializing data
import pandas as pd
catalouge = pd.read_csv('newData/flipkart_cleaned.csv')
purchase_history = pd.read_csv('newData/synthetic_v2.csv')

def get_recommendation(keywords_list): # getting the top 3 products based on keywords
    mask = catalouge['product_category_tree'].apply(lambda x: any(keyword in x for keyword in keywords_list))
    filtered = catalouge[mask]
    top_products = filtered.sort_values(by='overall_rating', ascending=False).head(3)

    # Formatting the output more clearly
    return "\n".join(
        f"**{idx + 1}. {row['product_name']}** - Discounted Price: {row['discounted_price']}, Description: {row['description']}"
        for idx, row in top_products.iterrows()
    )