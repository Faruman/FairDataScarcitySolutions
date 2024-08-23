# Use a pipeline as a high-level helper
from transformers import pipeline

messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe = pipeline("text-generation", model="unsloth/llama-3-8b-Instruct-bnb-4bit")
pipe(messages)