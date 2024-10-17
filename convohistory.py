import os
from dotenv import load_dotenv
from supabase import create_client, Client
from typing import List, Tuple

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_API_KEY = os.getenv("SUPABASE_API_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_API_KEY)

def add_chat_history_user(user_id, session_id, user_input, bot_response, user_intention):
    data = {
        "user_id": user_id,
        "session_id": session_id,  # save the session ID
        "user_input": user_input,
        "bot_response": bot_response,
        "intention": user_intention
    }
    response = supabase.table("chat_history").insert(data).execute()
    return response

def get_past_conversations_users(user_id, session_id):
    response = (supabase.table("chat_history")
                .select("intention")
                .eq("user_id", user_id)
                .eq("session_id", session_id)
                .order("created_at", desc=True) # sorting the conversations by the most recent interaction
                .limit(1) # getting the most recent interaction
                .execute())
    past_convo = response.data
    string = " ".join(d['intention'] for d in past_convo)

    return string



"""CONVERSATION HISTORY FOR GUEST USERS """
# function to update the chat history (which is stored in a list)
def add_chat_history_guest(session_id, user_input, bot_response, conversation_history):
    # Append the current user query and bot response as a tuple to the conversation history
    conversation_history.append((user_input, bot_response))


# function to get the past intention of the user
def get_past_conversation_guest(session_id, memory) -> List[Tuple[str, str]]:
    filtered_messages = [msg.content for msg in memory if msg.session_id == session_id]
    
    # Assume messages are strictly alternating between user query and bot response
    # Zip the messages pairwise: (query1, response1), (query2, response2), ...
    conversation_pairs = list(zip(filtered_messages[0::2], filtered_messages[1::2]))
    
    # Check if there is at least one pair and return the second item of the last pair
    if conversation_pairs:
        # Return the bot response of the last conversation pair
        return conversation_pairs[-1][1]
    else:
        # Return None if there are no conversation pairs
        return None