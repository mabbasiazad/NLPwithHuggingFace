from huggingface_hub import login
access_token_write = "abc" #read from the website
login(token = access_token_write)

from transformers import AutoModelForMaskedLM, AutoTokenizer

checkpoint = "camembert-base"

model = AutoModelForMaskedLM.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

model.push_to_hub("dummy")
tokenizer.push_to_hub("dummy")