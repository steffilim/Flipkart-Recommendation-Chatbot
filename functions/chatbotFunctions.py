import re
from functions.databaseFunctions import initialising_supabase
from recSys.weighted import hybrid_recommendations
import time

# Getting user intention
def getting_user_intention_dictionary(user_input, intention_chain, previous_intention, past_follow_up_questions,  items_recommended):
    """
    Extracts the user's intention as a dictionary using the provided intention chain.

    Args:
        user_input (str): The current user input.
        intention_chain (object): The chain used to process and extract user intention.
        previous_intention (dict): The previous user intention dictionary, if any.
        past_follow_up_questions (list): A list of past follow-up questions asked to the user.
        items_recommended (list): A list of items recommended to the user in the current session.

    Returns:
        dict: A dictionary representing the user's current intention.
    """

    if past_follow_up_questions is None:
        past_follow_up_questions = []

    user_intention_dictionary = intention_chain.invoke({"input": user_input, "previous_intention": previous_intention, "follow_up_questions": past_follow_up_questions, "items_recommended": items_recommended})

    return user_intention_dictionary

def get_item_details(db, product_index, session_id):
    """
    Fetches detailed information about a specific product from the database.

    Args:
        db (object): The MongoDB database connection object.
        product_index (int): The index of the product in the recommended items list.
        session_id (str): The unique identifier for the user's session.

    Returns:
        str: A human-readable string detailing the product's name, brand, price, rating, description, and specifications.
    """

    # Initialize Supabase connection
    supabase = initialising_supabase()
    
    # fetching from mongodb
    chat_session = db.chatSession
    chat_session_data = chat_session.find_one({"session_id": session_id}).get("message_list")
    last_recommendation = chat_session_data[-1].get("items_recommended")
    item_of_interest = last_recommendation[int(product_index)]

    # Get item of interest from mongodb
    readable_output = (
        f"Product Name: {item_of_interest['product_name']}\n"
        f"Brand: {item_of_interest['brand']}\n"
        f"Price: ₹{item_of_interest['discounted_price']}\n"
        f"Rating: {item_of_interest['overall_rating']}\n\n"
        f"Description: {item_of_interest['description']}\n\n"
        f"Product Details: {item_of_interest['product_specifications']}\n"
    )
    
    return readable_output

def get_item_details_guest(convo_history_list_guest, product_index):
    """
    Fetches detailed information about a specific product for a guest user.

    Args:
        convo_history_list_guest (list): The list of products recommended during the guest user's conversation history.
        product_index (int): The index of the product in the conversation history list.

    Returns:
        str: A human-readable string detailing the product's name, brand, price, rating, description, and specifications.
    """
    
    product_index = int(product_index)
    item_recommendation = convo_history_list_guest[product_index]

    readable_output = (
        f"Product Name: {item_recommendation['product_name']}\n"
        f"Brand: {item_recommendation['brand']}\n"
        f"Price: ₹{item_recommendation['discounted_price']}\n"
        f"Rating: {item_recommendation['overall_rating']}\n\n"
        f"Description: {item_recommendation['description']}\n\n"
        f"Product Details: {item_recommendation['product_specifications']}\n"
    )

    return readable_output

def getting_bot_response(user_intention_dictionary, chain2, supabase, db, previous_items_recommended, user_profile, user_purchases, user_id, session_id):
    """
    Generates the bot's response based on user intention, profile, and recommendations.

    Args:
        user_intention_dictionary (dict): A dictionary containing the user's intention details.
        chain2 (object): The chain used to process recommendations and follow-up questions.
        supabase (object): The initialized Supabase client instance.
        db (object): The MongoDB database connection object.
        previous_items_recommended (list): A list of items recommended in the current session.
        user_profile (dict): The user's profile information.
        user_purchases (list): A list of the user's purchase history.
        user_id (str): The unique identifier for the user.
        session_id (str): The unique identifier for the user's session.

    Returns:
        tuple: 
            - recommendations (list or None): A list of recommended products, or None if no recommendations are available.
            - bot_response (str): The response message generated for the user.
    """
  
    # Fetch the catalogue & users data from Supabase
    catalogue = (supabase.table('flipkart_cleaned_2k'))
    users_data = (supabase.table('synthetic_v2_2k').select('*').execute().data)

    item_availability = user_intention_dictionary.get("Available in Store")
    
    if item_availability == "No":
        print("Item not available")
        bot_response = user_intention_dictionary.get("Follow-Up Question")
        return None, bot_response
        
    # Related to Recommendation
    elif user_intention_dictionary.get("Related to Recommendation") == "Yes":
        
        product_id = user_intention_dictionary.get("Product ID")
        follow_up = user_intention_dictionary.get("Follow-Up Question")

        # Fetch item details
        if session_id == "guest":
            item_recommendation = get_item_details_guest(previous_items_recommended, product_id)
        else:
            item_recommendation = get_item_details(db, product_id, session_id)
        item_recommendation += "\n\n" + follow_up

        return None, item_recommendation

    else: 
        # Set fields_incomplete to 0 if "Fields Incompleted" is None or not in the dictionary
        fields_incomplete = int(user_intention_dictionary.get("Fields Incompleted",0))
        item = user_intention_dictionary.get("Product Item")         
        keen_to_share = user_intention_dictionary.get("Keen to Share")

        # Measure the time taken for hybrid_recommendations
        start_time = time.time()  # Record the start time
        
        # Check if all fields are incomplete and user prefers not to share more details
        if fields_incomplete == 3 and keen_to_share == "No":
            recommendations = hybrid_recommendations(user_intention_dictionary, user_id)

            # Set recommendations to None if it is an empty DataFrame
            if recommendations.empty:
                recommendations = None
            else:
                # Convert to dictionary format if recommendations is not empty
                recommendations = recommendations.to_dict(orient='records')
            
            questions = user_intention_dictionary.get("Follow-Up Question")
            bot_response = chain2.invoke({"recommendations": recommendations, "questions": questions, "user_profile": user_profile, "user_purchase_history": user_purchases})
           
        # Case where user has incomplete fields but is willing to share more preferences
        elif fields_incomplete == 3 and keen_to_share == "Yes":
            recommendations = None
            bot_response = user_intention_dictionary.get("Follow-Up Question")
            
        else:
            # Generate recommendations based on known preferences
            recommendations = hybrid_recommendations(user_intention_dictionary, user_id)

            # Set recommendations to None if it is an empty DataFrame
            if recommendations.empty:
                recommendations = None
            else:
                # Convert to dictionary format if recommendations is not empty
                recommendations = recommendations.to_dict(orient='records')

            # Getting follow-up questions from previous LLM if available
            questions = user_intention_dictionary.get("Follow-Up Question")
            bot_response = chain2.invoke({"recommendations": recommendations, "questions": questions,"user_profile": user_profile, "user_purchase_history": user_purchases})

            end_time = time.time()  # Record the end time
            elapsed_time = end_time - start_time  # Calculate the elapsed time
            print(f"Time taken for hybrid_recommendations: {elapsed_time:.4f} seconds")  # Print the elapsed time
        
    return recommendations, bot_response