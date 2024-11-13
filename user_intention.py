# Getting user intention
def getting_user_intention_dictionary(user_input, intention_chain, previous_intention, past_follow_up_questions):
    print("NEW FILE HELLOOOO")
    
    if past_follow_up_questions is None:
        past_follow_up_questions = []

    user_intention_dictionary = intention_chain.invoke({"input": user_input, "previous_intention": previous_intention, "follow_up_questions": past_follow_up_questions})

    return user_intention_dictionary