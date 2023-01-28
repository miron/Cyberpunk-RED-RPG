from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# Load the pre-trained model and tokenizer
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")

# Define the fine-tuning function
def fine_tune(model, tokenizer, dataset_path):
    with open("cyberpunk_lore.txt", "r") as f:
        dataset = f.read()

    # Tokenize the dataset and convert it to a tensor
    input_ids = tokenizer.encode(dataset, return_tensor='pt')

    # Fine-tune the model on the custom dataset
    model.fine_tune(input_ids)

# Fine-tune the model on the custom dataset
fine_tune(model, tokenizer, "cyberpunk_lore.txt")


# Use the fine-tuned model for generation
generated_text = model.generate(
    prompt="What's the slang used in Night City streets?")
print(generated_text)


