""" KEYWORD DETECTION FUNCTION """

import nltk
from rake_nltk import Rake
from nltk.corpus import words, wordnet

# Function to check if the user's input is valid
def is_valid_input(user_input, valid_user_ids, keywords):
    """
    Validates the user's input based on predefined user IDs and keywords.

    Args:
        user_input (str): The input provided by the user.
        valid_user_ids (list[str]): A list of valid user IDs.
        keywords (list[str]): A list of valid keywords.

    Returns:
        bool: True if the input contains valid tokens, False otherwise.
    """

    # Convert both user IDs and keywords to set for fast membership checking
    keywords = set(word.lower() for word in keywords)

    # Tokenize and validate
    tokens = nltk.word_tokenize(user_input)

    # Define validity check
    valid_tokens = [word for word in tokens if word.lower() in wordnet.words() or word in valid_user_ids or word.lower() in keywords]

    return len(valid_tokens) > 0

def extract_keywords(item):
    """
    Extracts ranked keywords from a given text using the Rake algorithm.

    Args:
        item (str): The text to extract keywords from.

    Returns:
        list[str]: A list of keywords ranked by relevance.
    """

    r = Rake()
    r.extract_keywords_from_text(item)
    query_keyword = r.get_ranked_phrases_with_scores()
    query_keyword_ls = [keyword[1] for keyword in query_keyword]
    return query_keyword_ls

def parse_user_intention(user_intention_dictionary):
    """
    Converts a string representation of a user intention dictionary into a Python dictionary.

    Args:
        user_intention_dictionary (str): The string representation of a dictionary.

    Returns:
        dict: A dictionary containing the parsed key-value pairs.
    """

    dictionary = {}
    lines = user_intention_dictionary.split("\n")
    current_key = None

    for line in lines:
        # Remove leading dashes and strip any extra whitespace from the line
        cleaned_line = line.lstrip('- ').strip()
        if ": " in cleaned_line:  # Check if the line has a colon and space, indicating a key-value pair
            key, value = cleaned_line.split(": ", 1)
            current_key = key.strip()
            dictionary[current_key] = value.strip()
        elif current_key:  # This line might be a continuation of the last key's value
            dictionary[current_key] += " " + line.strip()
    
    return dictionary