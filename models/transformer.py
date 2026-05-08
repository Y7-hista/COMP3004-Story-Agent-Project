import os
import re
import random
from dataclasses import dataclass
from typing import List, Dict, Optional, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class LLMGeneration:
    temperature: float = 0.85
    top_k: int = 50
    top_p: float = 0.92
    repetition_penalty: float = 1.15
    max_new_tokens: int = 170
    do_sample: bool = True

# class TransformerLLMModel:
#     def __init__(self, model_)