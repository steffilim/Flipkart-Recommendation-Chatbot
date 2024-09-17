import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

from langchain_core.output_parsers import StrOutputParser
from langchain.chains import SimpleSequentialChain, LLMChain

#from langchain_core.prompts.few_shot import FewShotPromptTemplate
import pandas as pd


# LLM initialisation
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
You are a friendly recommendation engine chatbot for an e-commerce. 
Your job is to come up with a product recommendation based on the user's query.
This is the user's query {input}. 
Your output should be in a form of a list of the main keywords that you think are relevant to the user's query.
If there are not enough keywords in the list to make a recommendation, you should ask the user for more information.
Else the users query is out of scope, you should inform the user that you are unable to help them and should seek help from a Live Agent instead.
"""
promptTemplate1 = PromptTemplate(input_variables = ['input'], template = template1)

chain1 = LLMChain(llm = llm, prompt = promptTemplate1)

template2 = """
You are a refined recommendation engine chatbot for an e-commerce.
Your job is to refine the output based off the input that has given to you. 
Here is the input that you have received {input}. 
If there is only one keyword in the list, you should ask the user for more information.
Else refine the recommendation. 
"""

promptTemplate2 = PromptTemplate(input_variables = ['input'], template = template2)
chain2 = LLMChain(llm = llm, prompt = promptTemplate2)



ssChain = SimpleSequentialChain(chains = [chain1, chain2],
                                verbose = True)


while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    result = ssChain.invoke(input = prompt)     # type(result) == dict of {input:..., output:...}
    print(result['output'])


