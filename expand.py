'''
This script takes in a topics file containing queries and outputs a new version of the topics file with the same format - containing an expanded Title field.
'''

# Import necessary libraries
import os
import re
from bs4 import BeautifulSoup
import json
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
import torch

def initialize_llama():
    '''initialize llama'''
    model_id = "meta-llama/Llama-3.2-1B"

    pipe = pipeline(
        "text-generation", 
        model=model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        max_length=100
    )

    return pipe

def initialize_gpt():
    '''initialize gpt'''
    
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    return model, tokenizer


def initialize_models(llm_model_name, llm_dir):
    """
    Initialize the LLaMA model.

    Parameters:
    llm_model_name (str): The name of the LLaMA model.
    llm_dir (str): The directory to save/load the LLaMA model.

    Returns:
    tuple: The tokenizer and LLaMA model.
    """
    # Check if the LLaMA model and tokenizer are already saved
    if not os.path.exists(llm_dir):
        os.makedirs(llm_dir)
        tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        llm_model = AutoModelForCausalLM.from_pretrained(llm_model_name)
        tokenizer.save_pretrained(llm_dir)
        llm_model.save_pretrained(llm_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(llm_dir)
        llm_model = AutoModelForCausalLM.from_pretrained(llm_dir)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    llm_model.to(device)

    return tokenizer, llm_model, device
    
# def expand_query_for_topic(tokenizer, llm_model, device, topic, role):
#     query_id = topic['Id']
#     title = topic['Title']
#     body = BeautifulSoup(topic['Body'], 'html.parser').get_text()
#     tags = ' '.join(topic['Tags'])
#     query = f"{title} {body} {tags}"
  
def gen_with_gpt2(model, tokenizer, text_input):
    '''Expand text input using gpt2'''
    
    input_ids = tokenizer(text_input, return_tensors="pt").input_ids

    gen_tokens = model.generate(
        input_ids,
        do_sample=True,
        temperature=0.9,
        max_length=100,
    )

    gen_text = tokenizer.batch_decode(gen_tokens)[0]
    return gen_text

def gen_with_llama(pipe, text_input):
    '''expand text using llama3.2 1b'''

    return pipe(text_input)[0]['generated_text']

def expand_query_title(model_mode, topic):
    '''Expand the title field of the topics json and return a copy of the json exactly identical to the original but with an expanded title
    args: 
    - model_mode: this can be either 'gpt2' or 'llama32' - it indicates which text generation pipeline to use.
    - topic: the json formatted dictionary containing queries with Id, Title, and Body fields
    '''

    return topic


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Expand queries using a bi-encoder model with query expansion.')
    
    # Add argument for the topics file
    parser.add_argument('--topics', type=str, required=True, help='The path to the topics file.')
    
    # Add argument for the LLM model name with restricted choices
    parser.add_argument('--llm_name', type=str, required=True, choices=['gpt2', 'llama32'], help='The LLM model to be used for query expansion.')

    # Add argument for the output file
    parser.add_argument('--output', type=str, required=True, help='The path to the output file.')
    
    # Parse the command-line arguments
    args = parser.parse_args()

    gen_mode = None

    if args.llm_name == 'gpt2':
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        gen_mode = 'gpt2'
    elif args.llm_name == 'llama32':
        model_id = "meta-llama/Llama-3.2-1B"

        pipe = pipeline(
            "text-generation", 
            model=model_id, 
            torch_dtype=torch.bfloat16, 
            device_map="auto",
            max_length=100
        )

        gen_mode = 'llama32'

    print(f"Generating expanded queries in the {gen_mode} execution mode.")

    print("Loading topics from file...")
    # Load the topics from the specified file
    with open(args.topics, 'r') as f:
        topics = json.load(f)

    print("Initializing models...")
    # Initialize the models
    tokenizer, llm_model, device = initialize_models('meta-llama/Llama-3.2-1B', args.llm_dir)

    print("Expanding queries...")
    expanded_queries = []
    

    
    print("Query expansion complete. Results saved to", args.output)