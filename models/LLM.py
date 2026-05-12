import os
import random
import re
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from models.topic_planner import TopicPlanner


class LLMModel:
    def __init__(self, model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0", local_model_path = "saved_models/llm", planner = None):
        # "google/gemma-3-1b-it", local_model_path = "./google/gemma-3-1b-it-local"
        # distilgpt2
        # TinyLlama/TinyLlama-1.1B-Chat-v1.0
        self.model_name = model_name
        self.local_model_path = local_model_path
        self.planner = planner

        self.tokenizer = None
        self.model = None
        self.temperature = 0.8
        self.repetition_penalty = 1.2
        self.top_k = 50
        self.top_p = 0.9
        self.max_length = 500

    def train(self, text = None, model_path = "saved_models/llm", temperature = 0.9, top_k =  50, top_p = 0.9, repetition_penalty = 1.2, max_length = 500):
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.max_length = max_length
        self.model_path = model_path

        if os.path.exists(self.local_model_path):
            print("Loading saved LLM")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
        else:
            print("Downloading LLM from HuggingFace")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

            os.makedirs(model_path, exist_ok = True)
            self.tokenizer.save_pretrained(model_path)
            self.model.save_pretrained(model_path)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = self.model.to("cpu")
        self.model.eval()


    def generate_sentence_llm(self, context_text, target, allow_dialogue=False):

        style = "Include a short dialogue using quotes." if allow_dialogue else "No dialogue."
        
        # Build prompts for texts (to improve the coherence, syntax)
        prompt = f"""
            You are a children's story writer.
            Write one short simple sentence.

            Topic: {target}

            Rules:
            - natural English
            - child-friendly
            - no explanations
            - no lists
            - no article writing
            - no code

            Sentence:
        """

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")

        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_new_tokens = 35,
                do_sample = True,
                temperature = 0.6,
                top_k = 40,
                top_p = 0.85,
                repetition_penalty = 1.2,
                eos_token_id = self.tokenizer.eos_token_id
            )
        # Decode 
        generated_ids = output[0][input_ids.shape[1]:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        text = re.sub(r"[^a-zA-Z ,.'\"-]", "", text)

        sentences = re.split(r'[.!?]', text)
        sentence = sentences[0].strip()

        if len(sentence.split()) < 5:
            return None

        if any(x in sentence for x in ["def", "return", "print", "#"]):
            return None

        return sentence.capitalize() + "."


    def generate_text(self, prompts):
        # Remove the generate_sentence (make it directly generate a entire story/texts)
        input_ids = self.tokenizer.encode(prompts, return_tensors = "pt")

        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_new_tokens = 120,   
                do_sample = True,
                temperature = 0.9,
                top_k = 40,
                top_p = 0.9,
                repetition_penalty = 1.1,
                eos_token_id = self.tokenizer.eos_token_id
            )
            generated_ids = output[0][input_ids.shape[1]:]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens = True)

            return text
        
    def clean_text(self, text):
        # Avoid some codes, points (make the output become a complete story)
        text = text.replace("\n", " ")
        text = re.sub(r"def .*", "", text)
        text = re.sub(r"final answer.*", "", text)
        text = re.sub(r"python.*", "", text)
        text = re.sub(r"\s+", " ", text).strip()

        sentences = re.split(r'(?<=[.!?])\s+', text)
        cleaned = []

        for s in sentences:
            if len(s.split()) > 4:
                cleaned.append(s)

        return " ".join(cleaned[:6])

    
    def generate_story(self, keywords, num_sentences=8):
        prompt = f"""
            You are a creative children's story writer.

            Write a short story.

            Keywords: {", ".join(keywords)}

            Requirements:
            - simple English
            - coherent story
            - child-friendly
            - include all keywords naturally
            - around {num_sentences} sentences
            - no bullet points
            - no explanations

            Story:
        """

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                attention_mask = attention_mask,
                max_new_tokens = 180,
                do_sample = True,
                temperature = 0.8,
                top_p = 0.92,
                repetition_penalty = 1.15,
                pad_token_id = self.tokenizer.eos_token_id,
                eos_token_id = self.tokenizer.eos_token_id
            )

        generated_ids = output[0][input_ids.shape[1]:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens = True)
        text = self.clean_text(text)

        return text