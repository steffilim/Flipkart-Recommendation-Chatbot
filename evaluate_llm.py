from prompt_template import intention_template
from gemini import parse_user_intention, intention_chain 
from fuzzywuzzy import fuzz
import json


test_cases = [
    # Test cases for evaluating the chat history and LLM response

    {
        "query": "I want a vase.",
        "previous_intention": "",
        "follow_up_questions": "",
        "items_recommended": [],
        "expected_user_intention": {  
            'Related to Follow-Up Questions': 'New',
            'Related to Recommendation': 'No',
            'Available in Store': 'Yes',
            'Brand': 'No preference',
            'Product Item': 'Vase',
            'Product Details': 'No preference',
            'Budget': 'No preference',
            'Keen to Share': 'Yes',
            'Fields Incompleted': '3',
            'To-Follow-Up': 'Yes',
            'Follow-Up Question': "I see you're interested in getting a vase. Could you please specify a brand, budget, or any other details? This will help me find the best options for you."
        }
    },
    {
        "query": "I'm looking for a necklace.",
        "previous_intention": "",
        "follow_up_questions": "",
        "items_recommended": [],
        "expected_user_intention": {
            'Related to Follow-Up Questions': 'New',
            'Related to Recommendation': 'No',
            'Available in Store': 'Yes',
            'Brand': 'No preference',
            'Product Item': 'Necklace',
            'Product Details': 'No preference',
            'Budget': 'No preference',
            'Keen to Share': 'Yes',
            'Fields Incompleted': '3',
            'To-Follow-Up': 'Yes',
            'Follow-Up Question': "I see you're interested in getting a necklace. Could you please specify a brand, budget, or any other details? This will help me find the best options for you."
        }
    },
    {
        "query": "My budget is 3000.",
        "previous_intention": "Looking for a necklace",
        "follow_up_questions": "I see you're interested in getting a necklace. Could you please specify a brand, budget, or any other details? This will help me find the best options for you.",
        "items_recommended": [],
        "expected_user_intention": {
            'Related to Follow-Up Questions': 'Old',
            'Related to Recommendation': 'No',
            'Available in Store': 'Yes',
            'Brand': 'No preference',
            'Product Item': 'Necklace',
            'Product Details': 'No preference',
            'Budget': '3000',
            'Keen to Share': 'Yes',
            'Fields Incompleted': '2',
            'To-Follow-Up': 'Yes',
            'Follow-Up Question': "Could you please specify a brand or any specific features you are looking for in a necklace?"
        }
    },
    {
        "query": "I want it in silver",
        "previous_intention": "Looking for a necklace with budget of 3000",
        "follow_up_questions": "Could you please specify a brand or any specific features you are looking for in a necklace?",
        "items_recommended": [],
        "expected_user_intention": {
            'Related to Follow-Up Questions': 'Old',
            'Related to Recommendation': 'No',
            'Available in Store': 'Yes',
            'Brand': 'No preference',
            'Product Item': 'Necklace',
            'Product Details': 'Silver',
            'Budget': '3000',
            'Keen to Share': 'Yes',
            'Fields Incompleted': '1',
            'To-Follow-Up': 'Yes',
            'Follow-Up Question': "Could you please specify a brand you prefer?"
        }
    },
    {
        "query": "I would prefer the brand Swarovski",
        "previous_intention": "Looking for a silver necklace with budget of 3000",
        "follow_up_questions": "Could you please specify a brand you prefer?",
        "items_recommended": [],
        "expected_user_intention": {
            'Related to Follow-Up Questions': 'Old',
            'Related to Recommendation': 'No',
            'Available in Store': 'Yes',
            'Brand': 'Swarovski',
            'Product Item': 'Necklace',
            'Product Details': 'Silver',
            'Budget': '3000',
            'Keen to Share': 'Yes',
            'Fields Incompleted': '0',
            'To-Follow-Up': 'No',
            'Follow-Up Question': "Do the options presented meet your requirements, or would you like to explore other products?"
        }
    },
    {
        "query": "hi im looking for running shoes",
        "previous_intention": "",
        "follow_up_questions": "",
        "items_recommended": [],
        "expected_user_intention": {
            'Related to Follow-Up Questions': 'New',
            'Related to Recommendation': 'No',
            'Available in Store': 'Yes',
            'Brand': 'No preference',
            'Product Item': 'running shoes',
            'Product Details': 'No preference',
            'Budget': 'No preference',
            'Keen to Share': 'Yes',
            'Fields Incompleted': '3',
            'To-Follow-Up': 'No',
            'Follow-Up Question': "I see you're interested in getting running shoes. Could you please specify a brand, budget, or any other details? This will help me find the best options for you."
        }
    }
    ,
    {
        "query": "my preferred brand is atis amco",
        "previous_intention": "Looking for running shoes",
        "follow_up_questions": "I see you're interested in getting running shoes. Could you please specify a brand, budget, or any other details? This will help me find the best options for you.",
        "items_recommended": [],
        "expected_user_intention": {
            'Related to Follow-Up Questions': 'Old',
            'Related to Recommendation': 'No',
            'Available in Store': 'Yes',
            'Brand': 'atis amco',
            'Product Item': 'running shoes',
            'Product Details': 'No preference',
            'Budget': 'No preference',
            'Keen to Share': 'Yes',
            'Fields Incompleted': '2',
            'To-Follow-Up': 'No',
            'Follow-Up Question': "Could you please specify a budget range or any specific features you need in your running shoes?"
        }
    }
]


def get_llm_intention(test_case):
    """
    Retrieves the LLM output by invoking the intention chain for a test case.

    Args:
        test_case (dict): A dictionary containing user input, previous intentions, and expected results.

    Returns:
        str: The LLM's response as a string.
    """

    query = test_case["query"]
    previous_intention = test_case["previous_intention"]
    follow_up_questions = test_case["follow_up_questions"]
    items_recommended = test_case["items_recommended"]

    # Create prompt variables
    prompt_variables = {
        "input": query,
        "previous_intention": previous_intention,
        "follow_up_questions": follow_up_questions,
        "items_recommended": items_recommended,
    }

    # Generate LLM output
    llm_output = intention_chain.invoke(prompt_variables)
    return llm_output

def evaluate_intention(test_case, llm_function):
    """
    Evaluates the generated LLM output against the expected output for a given test case.

    Args:
        test_case (dict): A dictionary containing the test case details.
        llm_function (function): A function that generates the LLM output.

    Returns:
        Tuple: A tuple containing a success flag and feedback string.
    """

    llm_output = llm_function(test_case)
    
    # Convert the llm_output into a dictionary for comparison
    llm_output_dict = parse_user_intention(llm_output)
    
    # Expected output from test case
    expected_output = test_case["expected_user_intention"]
    
    # Ensure both are displayed in a single-line dictionary format
    formatted_expected_output = json.dumps(expected_output, indent=None)
    formatted_llm_output = json.dumps(llm_output_dict, indent=None)
    
    # Similarity score between the outputs
    similarity_score = fuzz.token_set_ratio(formatted_llm_output, formatted_expected_output)
    
    # Add meaningful feedback with new lines
    feedback = (
        f"Expected:\n{formatted_expected_output}\n\n"
        f"LLM Generated:\n{formatted_llm_output}\n\n"
        f"Similarity: {similarity_score}%"
    )
    success = similarity_score > 90 # If similarity score > 90, considered successful
    return success, feedback

def run_evaluation():
    """
    Runs the evaluation for all test cases and prints the results.

    Returns:
        None
    """
        
    for idx, test_case in enumerate(test_cases, start=1):
        print(f"Test Case {idx}:")
        success, feedback = evaluate_intention(test_case, get_llm_intention)
        print(feedback)
        print("\n" + "="*50 + "\n")  

if __name__ == "__main__":
    run_evaluation()