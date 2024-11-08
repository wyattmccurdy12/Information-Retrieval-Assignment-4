# Ranking and Re-Ranking System

This project implements a ranking and re-ranking system using bi-encoder and cross-encoder models. Both pretrained and finetuned models are utilized.

## Project Structure

- `main.ipynb`: Jupyter notebook for interactive development and experimentation.
- `main.py`: Main script for running the ranking and re-ranking system.
- `README.md`: Project documentation.

## Models

### Bi-Encoder

The bi-encoder model is used for initial ranking. It encodes queries and documents independently.

### Cross-Encoder

The cross-encoder model is used for re-ranking. It takes pairs of queries and documents and encodes them together for more accurate scoring.

## Usage

### Running the Script

To run the script, you may run the `batch_retrieve.sh` script on the command line. It contains command line arguments with all flag combinations. 

## Command-Line Arguments

The `main.py` script accepts several command-line arguments to configure the ranking and re-ranking system. Below is a description of each argument:

- `-r`, `--rerank`: Perform re-ranking (optional flag).
- `-q`, `--queries`: Path to the queries file (required).
- `-d`, `--documents`: Path to the documents file (required).
- `-be`, `--bi_encoder`: Bi-encoder model string (required).
- `-ce`, `--cross_encoder`: Cross-encoder model string (required).
- `-ft`, `--finetuned`: Indicate if the model is fine-tuned (optional flag).

Output filenames will be generated based on the names of the inputs and the models used.