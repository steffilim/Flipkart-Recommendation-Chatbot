from fastapi import FastAPI
from pydantic import BaseModel
from gemini import ssChain, to_list, get_recommendation  # Import your chatbot logic

app = FastAPI()

# Request body model
class UserQuery(BaseModel):
    message: str

# Define your route for chatbot interaction
@app.post("/chat")
async def chat(user_query: UserQuery):
    prompt = user_query.message
    intermediate_results = ssChain.invoke(input=prompt)
    results_ls = to_list(intermediate_results['query'])

    if len(results_ls) <= 1:  # no recommendations found
        refined_response = ssChain.invoke(input=prompt)
        return {"response": refined_response['refined']}
    else:
        recommendations = get_recommendation(results_ls)
        refined_response = ssChain.invoke(input=recommendations)
        return {"response": refined_response['refined']}