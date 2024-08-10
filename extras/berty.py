from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("Clinical-AI-Apollo/Medical-NER")
model = AutoModelForTokenClassification.from_pretrained("Clinical-AI-Apollo/Medical-NER")

# Input text
text = "my woman"

# Tokenize the input text
inputs = tokenizer(text, return_tensors="pt", truncation=True)

# Perform prediction
with torch.no_grad():
    outputs = model(**inputs)

# Get the predicted token classes
predictions = torch.argmax(outputs.logits, dim=-1)

# Decode the token classes back to words
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
labels = [model.config.id2label[pred.item()] for pred in predictions[0]]
print([label[2:] for label in labels if label != 'O'])
# Combine tokens and labels
entities = []
for token, label in zip(tokens, labels):
    if label != 'O':  # Ignore 'O' (non-entity) labels
        entities.append((token, label))

# Print the recognized entities
for token, label in entities:
    print(f"{token}: {label}")
