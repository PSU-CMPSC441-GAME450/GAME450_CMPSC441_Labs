import torch
import transformers

tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')
model = transformers.GPT2LMHeadModel.from_pretrained('gpt2')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def generate_journal_entry(city_name):
    prompt = "I arrived in " + city_name + " today. "
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    output = model.generate(input_ids, max_length=200, do_sample=True)
    journal_entry = tokenizer.decode(output[0], skip_special_tokens=True)
    return journal_entry
