""" DATABASE FUNCTION"""

# Initializing data
import pandas as pd
from pymongo import MongoClient
import pymongo
import os
from datetime import datetime, timedelta
from forex_python.converter import CurrencyCodes
from dotenv import load_dotenv
from supabase import create_client
import ast

currency = CurrencyCodes()
INR = currency.get_symbol('INR')

def initialising_supabase():
    """
    Initializes the Supabase client using environment variables.

    Environment Variables:
        SUPABASE_URL (str): The URL of the Supabase project.
        SUPABASE_API_KEY (str): The API key for accessing the Supabase project.

    Returns:
        supabase.Client: An initialized Supabase client instance.
    """

    load_dotenv()
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_API_KEY = os.getenv("SUPABASE_API_KEY")
    supabase = create_client(SUPABASE_URL, SUPABASE_API_KEY)
    return supabase

def initialising_mongoDB():
    load_dotenv()
    MONGODB_URI = os.getenv("MONGODB_URI")
    FLIPKART = os.getenv("FLIPKART")
    client = pymongo.MongoClient(MONGODB_URI)
    mydb = client[FLIPKART]
    return mydb

def load_product_data(supabase=None):
    """
    Initializes the MongoDB client and connects to the specified database.

    Environment Variables:
        MONGODB_URI (str): The connection URI for MongoDB.
        FLIPKART (str): The name of the MongoDB database to connect to.

    Returns:
        pymongo.database.Database: A MongoDB database instance.
    """

    if supabase is None:
        supabase = initialising_supabase()
    # Load data from the flipkart_cleaned table in supabase
    catalogue_data = supabase.table('flipkart_cleaned_2k').select('*').execute().data
 
    return catalogue_data

def load_order_data(supabase=None): 
    """
    Loads product data from the 'flipkart_cleaned_2k' table in Supabase.

    Args:
        supabase (supabase.Client, optional): An initialized Supabase client. 
                                              If not provided, a new client is initialized.

    Returns:
        list[dict]: A list of dictionaries containing product data from the Supabase table.
    """
    
    if supabase is None:
        supabase = initialising_supabase()
    users_data = pd.DataFrame(supabase.table('synthetic_v2_2k').select('*').execute().data)
    return users_data