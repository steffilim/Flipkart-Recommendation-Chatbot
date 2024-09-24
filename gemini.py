import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import SequentialChain, LLMChain

import pandas as pd
from convohistory import add_chat_history   

# LLM INITIALISATION
# authenticating model
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(
        model="gemini-pro", 
        google_api_key=google_api_key,
        temperature=0.2, 
        verbose = True, 
        stream = True
    )


template1 = """
You are a smart LLM for an e-commerce company.
Imagine that you are selling the same products as Amazon.com
You are to identify the keywords in the query of the user. 
Here is the input that you have received {question}.

If the user is asking for a product that cannot be found in the e-commerce company, you should tell them that you are not able to help and should either search for something else or ask for customer service. 
The output for the list should be "None".
Else, identify the important keywords in the query and put them in a list and separate the keywords with a comma.

"""



template2 = """
You are a refined recommendation engine chatbot for an e-commerce online company.
Your job is to refine the output based off the input that has given to you. 

If you have received a list with the word "None", you should tell the user that you are not able to help and should either search for something else or ask for customer service.


Else:
You have been given a list of recommendations with the following headers: Product Name, Price, Description, User Interests.
Make sure the recommendations align with the user's interests, if possible. 
Summarise the product description. 
Omit the product number and give it in the following format:
1. **Product Name** : Price, 
 - Description

Here is the user's question :{query}
You should ask the user if the provided recommendations suit their needs or if they want another set of recommendations. 

"""


# DATA INITIALISATION
# initialising data
catalouge = pd.read_csv('newData/flipkart_cleaned.csv')
purchase_history = pd.read_csv('newData/synthetic_v2.csv')

# creating a sample recommender system

def get_recommendation(keywords_list): # getting the top 3 products based on keywords
    mask = catalouge['product_category_tree'].apply(lambda x: any(keyword in x for keyword in keywords_list))
    filtered = catalouge[mask]

    # Get the user's purchase history
    user_purchases = purchase_history[purchase_history['User ID'] == user_id]
    user_interests = user_purchases['User Interests'].iloc[0]

    # Filter out products that the user has already purchased
    purchased_product_ids = user_purchases['Product ID'].unique()
    filtered = filtered[~filtered['uniq_id'].isin(purchased_product_ids)]

    if user_interests:
        filtered['interest_match'] = filtered['product_category_tree'].apply(lambda x: any(interest in x for interest in user_interests.split(',')))
        filtered = filtered.sort_values(by=['interest_match', 'overall_rating'], ascending=[False, False])

    top_products = filtered.sort_values(by='overall_rating', ascending=False).head(3)

    # Formatting the output more clearly
    return "\n".join(
        f"**{idx + 1}. {row['product_name']}** - Discounted Price: {row['discounted_price']}, Description: {row['description']}"
        for idx, row in top_products.iterrows()
    )



# CHAINING

# chaining the recommendation system
promptTemplate1 = PromptTemplate(input_variables = ["question"], template = template1)
chain1 = LLMChain(llm = llm,
                  output_key = "query",
                  prompt = promptTemplate1)


promptTemplate2 = PromptTemplate(input_variables = ["query"], template = template2)

chain2 = LLMChain(llm = llm,
                  output_key = "refined",
                  prompt = promptTemplate2, 
                  verbose = True)

ssChain = SequentialChain(chains = [chain1, chain2],
                                input_variables = ["question"],
                                output_variables = ["refined"],
                                verbose = True)

def to_list(text):
    return text.split(',')



while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    user_id = input("Enter your User ID: ") # We need this for personalisation
    intermediate_results = chain1.invoke(input = prompt)
    results_ls = to_list(intermediate_results['query'])
    print(results_ls)
    if len(results_ls) <= 1: # no recommendations found
        print("NEED MORE INFORMATION")
        result = ssChain.invoke(input = prompt)
        print(result['refined'])
        
        # Store conversation history
        add_chat_history(prompt, result['refined']) #Call the function to add convo history into database
        continue
    else:
        print("RECOMMENDATION FOUND")
        recommendations = get_recommendation(results_ls)

    # Invoke the second chain for refining recommendations
        result = chain2.invoke(input=recommendations)
        print(result['refined'])
        # Assuming 'result' is a dictionary returned with refined recommendations
        #print(result['query'])
        #print(intermediate_results['text'])

        add_chat_history(prompt, result['refined']) #Call the function to add convo history into database
        continue