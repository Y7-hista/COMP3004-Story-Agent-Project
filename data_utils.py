import re

def tokenize(text):
    text = text.lower()
    tokens = text.split()
    return ["<START>"] + tokens + ["<END>"]