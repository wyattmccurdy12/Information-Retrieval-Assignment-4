# Import necessary libraries
import os
import re
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import json
import argparse
from tqdm import tqdm
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed

class Retriever:
    """
    A class to represent a retriever model which can be a bi-encoder.

    Attributes:
    model_name (str): The name of the bi-encoder model.
    model (SentenceTransformer): The bi-encoder model.
    device (torch.device): The device to run the model on (CPU or GPU).
    """

    def __init__(self, model_name):
        """
        The constructor for Retriever class.

        Parameters:
        model_name (str): The name of the bi-encoder model.
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def retrieve(self, query, documents, batch_size=32):
        """
        Retrieves documents relevant to the query.

        Parameters:
        query (str): The query string.
        documents (list): The list of documents.
        batch_size (int): The batch size for encoding documents.

        Returns:
        list: The list of relevant documents.
        """
        query_embedding = self.model.encode(query, convert_to_tensor=True).to(self.device)
        doc_texts = [doc['Text'] for doc in documents]
        
        # Encode documents in batches
        doc_embeddings = []
        for i in range(0, len(doc_texts), batch_size):
            batch_texts = doc_texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch_texts, convert_to_tensor=True).to(self.device)
            doc_embeddings.append(batch_embeddings)
        doc_embeddings = torch.cat(doc_embeddings, dim=0)
        
        distances = torch.norm(doc_embeddings - query_embedding, dim=1)
        sorted_indices = torch.argsort(distances)
        return [documents[i] for i in sorted_indices[:100].cpu().numpy()]  # Return top 100 documents

def process_topic(retriever, topic, documents):
    query_id = topic['Id']
    title = topic['Title']
    body = BeautifulSoup(topic['Body'], 'html.parser').get_text()
    tags = ' '.join(topic['Tags'])
    query = f"{title} {body} {tags}"
    
    # Retrieve relevant documents using the query
    relevant_documents = retriever.retrieve(query, documents)
    
    # Format the results in TREC format
    results = []
    for rank, doc in enumerate(relevant_documents):
        doc_id = doc['Id']
        score = 1.0 / (rank + 1)  # Example scoring function
        results.append(f"{query_id}\tQ0\t{doc_id}\t{rank + 1}\t{score}\tmy_run\n")
    
    return results

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Retrieve documents using a bi-encoder model.')
    
    # Add argument for the topics file
    parser.add_argument('--topics', type=str, required=True, help='The path to the topics file.')
    
    # Add argument for the path to the documents file
    parser.add_argument('--documents', type=str, required=True, help='The path to the documents file.')
    
    # Add argument for the output file
    parser.add_argument('--output', type=str, required=True, help='The path to the output file.')
    
    # Parse the command-line arguments
    args = parser.parse_args()

    print("Loading topics from file...")
    # Load the topics from the specified file
    with open(args.topics, 'r') as f:
        topics = json.load(f)

    print("Loading documents from file...")
    # Load the documents from the specified file
    with open(args.documents, 'r') as f:
        documents = json.load(f)

    print("Initializing retriever model...")
    # Create an instance of the Retriever class with the specified bi-encoder model
    retriever = Retriever('sentence-transformers/all-MiniLM-L6-v2')

    print("Processing topics and retrieving documents...")
    results = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_topic, retriever, topic, documents) for topic in topics]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing topics"):
            results.extend(future.result())

    # Save the results to the output file
    with open(args.output, 'w') as output_file:
        output_file.writelines(results)

    print("Document retrieval complete. Results saved to", args.output)