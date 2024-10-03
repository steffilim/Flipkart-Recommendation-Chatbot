import os
#import psycopg2
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_API_KEY = os.getenv("SUPABASE_API_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_API_KEY)

def add_chat_history(user_id, user_input, bot_response, user_intention):
    data = {
        "user_id": user_id,
        "user_input": user_input,
        "bot_response": bot_response,
        "intention": user_intention
    }
    response = supabase.table("chat_history").insert(data).execute()
    return response

def get_past_conversations(user_id):
    response = supabase.table("chat_history").select("*").eq("user_id", user_id).execute()
    return response.data
 