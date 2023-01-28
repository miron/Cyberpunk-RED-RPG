import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the pre-trained model and tokenizer
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Open the text file containing the custom dataset
with open("cyberpunk_lore.txt", "r") as f:
    dataset = f.readlines()

# Tokenize the dataset and convert it to a tensor
input_ids = torch.tensor([tokenizer.encode(text) for text in dataset])

# Fine-tune the model on the custom dataset
model.train()
model.fit(input_ids)

