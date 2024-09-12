import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# initialising model
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(
        model="gemini-pro", 
        google_api_key=google_api_key,
        temperature=0.5, 
        verbose = True, 
        stream = True
    )

prompt = PromptTemplate.from_template(
    'You are a recommendation chatbot for an e-commerce company that assist customers in finding the best products for their needs.'
    'The user query is "{query}".'
    'If the user asks a question that is unrelated to finding for a product, apologise and tell them that you are not able to help and customer service will be able to help you.'
)

chain = prompt | llm | StrOutputParser()
while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    result = chain.invoke(input = prompt)
    print(result)


