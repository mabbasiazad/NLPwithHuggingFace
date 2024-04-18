# useful link: https://huggingface.co/learn/nlp-course/chapter3/2?fw=pt
# the task is getting a pair a sentences and determines wether these 
# two sentences are related to each other or not

from transformers import AutoTokenizer

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


from datasets import load_dataset

raw_datasets = load_dataset("glue", "mrpc")
print(raw_datasets)

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True, max_length=128)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
print(tokenized_datasets)
print(tokenized_datasets.column_names)

# remove columns the model does not expect to see
tokenized_datasets = tokenized_datasets.remove_columns(["idx", "sentence1", "sentence2"])
#the model expects the argument to be named labels
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
#pytorch tensors instead of lists
tokenized_datasets = tokenized_datasets.with_format("torch")

tokenized_datasets["train"].column_names
# the model just accept these columns
# ["attention_mask", "input_ids", "labels", "token_type_ids"]

from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader

data_collator = DataCollatorWithPadding(tokenizer)
train_dataloader = DataLoader(tokenized_datasets["train"], batch_size=16, 
                              shuffle=True, collate_fn=data_collator)

for step, batch in enumerate(train_dataloader): 
    print(batch["input_ids"].shape)
    if step > 5:
        break
