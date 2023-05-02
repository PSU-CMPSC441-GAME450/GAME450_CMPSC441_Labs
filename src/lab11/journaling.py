import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_journal_entry(city_name):
    prompt = f"I arrived in {city_name} today. "
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    output = model.generate(input_ids=input_ids, max_length=200, do_sample=True)
    journal_entry = tokenizer.decode(output[0], skip_special_tokens=True)
    return journal_entry