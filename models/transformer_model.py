# models/transformer_model.py

from transformers import pipeline

generator = pipeline("text-generation", model="distilgpt2")

def generate_with_transformer(prompt):
    output = generator(prompt, max_length=100, num_return_sequences=1)
    return output[0]["generated_text"]