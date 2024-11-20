import re
from functions.databaseFunctions import initialising_supabase
from recSys.weighted import hybrid_recommendations

# Getting user intention
def getting_user_intention_dictionary(user_input, intention_chain, previous_intention, past_follow_up_questions,  items_recommended):
    if past_follow_up_questions is None:
        past_follow_up_questions = []

    
    user_intention_dictionary = intention_chain.invoke({"input": user_input, "previous_intention": previous_intention, "follow_up_questions": past_follow_up_questions, "items_recommended": items_recommended})

    return user_intention_dictionary

def get_item_details(db, product_index, session_id):
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
    product_index = int(product_index)
    item_recommendation = convo_history_list_guest[product_index]
    print("line 283: ", item_recommendation)

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
        print("line 305: ", product_id)
        follow_up = user_intention_dictionary.get("Follow-Up Question")

        #print(product_id)
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
           
    return recommendations, bot_response