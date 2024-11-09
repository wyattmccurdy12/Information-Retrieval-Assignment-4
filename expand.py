'''
Wyatt McCurdy
11/8/2024
Part of homework 4 for Information Retrieval

This script takes in a topics file containing queries and outputs a new version of 
the topics file with the same format - containing an expanded Title field.
'''

# Import necessary libraries
import os
import re
from bs4 import BeautifulSoup
import json
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

def initialize_llama():
    '''Initialize the LLaMA model pipeline.'''
    model_id = "meta-llama/Llama-3.2-1B"

    pipe = pipeline(
        "text-generation", 
        model=model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        max_length=200  # Set max_length instead of max_new_tokens
    )

    return pipe

def initialize_gpt():
    '''Initialize the GPT-2 model and tokenizer.'''
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    return model, tokenizer

def gen_with_gpt2(model, tokenizer, text_input):
    '''Expand text input using GPT-2.'''
    max_length = 200
    inputs = tokenizer.encode(text_input, return_tensors="pt")
    if inputs.size(1) > max_length:
        print(f"Skipping input as it exceeds max length of {max_length} tokens.")
        return text_input  # Return the original text if it exceeds max length
    inputs = inputs.to(model.device)
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1)
    gen_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return gen_text

def gen_with_llama(pipe, text_input):
    '''Expand text input using LLaMA 3.2 1B.'''
    gen_text = pipe(text_input, max_length=200)[0]['generated_text']
    return gen_text

def expand_query_title(model_mode, topic, model, tokenizer_or_pipe):
    '''Expand the title field of the topics JSON and return a copy of the JSON exactly identical to the original but with an expanded title.
    
    Args: 
    - model_mode: This can be either 'gpt2' or 'llama32' - it indicates which text generation pipeline to use.
    - topic: The JSON formatted dictionary containing queries with Id, Title, and Body fields.
    - model: The model used for text generation.
    - tokenizer_or_pipe: The tokenizer or pipeline used for text generation.
    
    Returns:
    - topic: The updated topic with an expanded title.
    '''
    title = topic['Title']
    body = BeautifulSoup(topic['Body'], 'html.parser').get_text()
    tags = ' '.join(topic['Tags'])
    text_input = f"{title} {body} {tags}"

    if model_mode == 'gpt2':
        expanded_title = gen_with_gpt2(model, tokenizer_or_pipe, text_input)
    elif model_mode == 'llama32':
        expanded_title = gen_with_llama(tokenizer_or_pipe, text_input)

    topic['Title'] = expanded_title
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
    model = None
    tokenizer_or_pipe = None

    # Initialize the appropriate model and tokenizer/pipeline based on user input
    if args.llm_name == 'gpt2':
        model, tokenizer = initialize_gpt()
        tokenizer_or_pipe = tokenizer
        gen_mode = 'gpt2'
    elif args.llm_name == 'llama32':
        tokenizer_or_pipe = initialize_llama()
        gen_mode = 'llama32'

    print(f"Generating expanded queries in the {gen_mode} execution mode.")

    print("Loading topics from file...")
    # Load the topics from the specified file
    with open(args.topics, 'r') as f:
        topics = json.load(f)

    print("Expanding queries...")
    expanded_queries = []
    
    # Iterate over each topic with a progress bar
    for topic in tqdm(topics, desc="Expanding queries"):
        expanded_query = expand_query_title(gen_mode, topic, model, tokenizer_or_pipe)
        expanded_queries.append(expanded_query)
    
    print("Saving expanded queries to output file...")
    # Save the expanded queries to the output JSON file
    with open(args.output, 'w') as output_file:
        json.dump(expanded_queries, output_file, indent=4)
    
    print("Query expansion complete. Results saved to", args.output)