import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Select device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("Using device:", device)

# Load model and tokenizer directly from Hugging Face
model_name = "roneneldan/TinyStories-1M"  # similar small model to TinyStories-1L-21M
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

model.eval()
print("Model loaded successfully!")

# Input text
prompt = "Hello World"

# Tokenize input
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Forward pass with activations captured
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True, return_dict=True)

logits = outputs.logits
hidden_states = outputs.hidden_states  # tuple of activations from each layer

print("=== Model Output ===")
print("Logits shape:", logits.shape)
print("Hidden states (activations):", len(hidden_states), "layers")
print("Last hidden state shape:", hidden_states[-1].shape)

# Optionally save results
torch.save(logits.cpu(), "logits.pt")
torch.save(hidden_states[-1].cpu(), "activations.pt")

print("Objects successfully saved to logits.pt and activations.pt!")