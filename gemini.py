import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import SimpleSequentialChain, LLMChain

import pandas as pd


# LLM INITIALISATION
# authenticating model
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY2")
llm = ChatGoogleGenerativeAI(
        model="gemini-pro", 
        google_api_key=google_api_key,
        temperature=0.2, 
        verbose = True, 
        stream = True
    )


template1 = """
You are a smart LLM that identifies the keywords in the query of the user. 
Here is the input that you have received {input}.
Identify the important keywords in the query and put them in a list. 
Separate the keywords with a comma.

"""



template2 = """
You are a refined recommendation engine chatbot for an e-commerce.
Your job is to refine the output based off the input that has given to you. 
Here is the input that you have received {input}. 
If there is only one keyword in the list, you should ask the user for more information.
Else refine the recommendation. 
"""


# DATA INITIALISATION
# initialising data
catalouge = pd.read_csv('newData/flipkart_cleaned.csv')

# creating a recommender system
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel

tfv = TfidfVectorizer(
    max_features=None,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',  # Use a raw string here
    ngram_range=(1,3),
    stop_words='english'
)


# fitting the description column
tfv_matrix = tfv.fit_transform(catalouge['description'])

# sigmoid kernel
sig = sigmoid_kernel(tfv_matrix, tfv_matrix)

# indices
indices = pd.Series(catalouge.index, index = catalouge['product_name']).drop_duplicates()


def get_recommendation(title, sig = sig):
    indx = indices[title]

    # getting sigmoid scores
    sig_scores = list(enumerate(sig[indx]))

    # sorting the items
    sig_scores = sorted(sig_scores, key = lambda x: x[1], reverse = True)

    # getting the top 10 items
    sig_scores = sig_scores[1:11]
    product_indices = [i[0] for i in sig_scores]
    return catalouge['product_name'].iloc[product_indices]


# CHAINING

# chaining the recommendation system
promptTemplate1 = PromptTemplate(input_variables = ['input'], template = template1)
chain1 = LLMChain(llm = llm, prompt = promptTemplate1)


promptTemplate2 = PromptTemplate(input_variables = ['input'], template = template2)
chain2 = LLMChain(llm = llm, prompt = promptTemplate2)
ssChain = SimpleSequentialChain(chains = [chain1, chain2],
                                verbose = True)

def to_list(text):
    return text.split(',')



while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    intermediate_results = chain1.invoke(input = prompt)
    results_ls = to_list(intermediate_results['text'])
    print(results_ls)
    if len(results_ls) <= 1: # no recommendations found
        print("NEED MORE INFORMATION")
        result = ssChain.invoke(input = prompt)
        print(result['output'])
        continue
    else:
        print("RECOMMENDATION FOUND")
        recommendations = get_recommendation(results_ls)
        print(intermediate_results['text'])
        continue


   


