
"""
Pre-requisite: run data_preparation.py first
Project: Condition Classifier
train a classifier that can predict the patient condition based on the drug review.
"""
import torch
# loading the dataset
from datasets import load_from_disk

drug_dataset = load_from_disk("Projects/ConditionPrediction/drug-reviews-data")
print(drug_dataset)

# Tokenizing the dataset

condition_labels = []
for split in drug_dataset.keys():
    print(split)
    condition_labels = drug_dataset[split]["condition"]
    condition_labels = condition_labels
    condition_labels.extend(condition_labels)

condition_labels = list(set(condition_labels))  
print(len(condition_labels))

# Define the str_to_int dictionary
label2id = {label: i for i, label in enumerate(condition_labels)}

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# condition_labels_ = []
def tokenize_function(example):
    tokenized_example = tokenizer(example["review"], truncation=True) 
    return tokenized_example

tokenized_datasets = drug_dataset.map(tokenize_function, batched=True)
print(tokenized_datasets)


# print(len(list(set(condition_labels_))))
# str_to_int = {condition: i for i, condition in enumerate(condition_labels_)}

# def create_label(example):
#     example["labels"] = [str_to_int[cond] for cond in example["condition"]] 
#     return example

# tokenized_datasets = tokenized_datasets.map(create_label)
# print(tokenized_datasets)

tokenized_datasets = tokenized_datasets.remove_columns(['patient_id', 'drugName', 'review', 'rating', 'date', 'usefulCount', 'review_length'])
# tokenized_datasets = tokenized_datasets.rename_column("condition", "labels")
# tokenized_datasets.set_format("torch")
print(tokenized_datasets["train"][0])


condition_list = []
for split in tokenized_datasets.keys():
    print(split)
    conditions = tokenized_datasets[split]["condition"]   
    condition_list.extend(conditions)

condition_list = list(set(condition_list))  
print(len(condition_list))

# Define the str_to_int dictionary
label2id = {label: i for i, label in enumerate(condition_list)}
id2label = {i: label for label, i in label2id.items()}

def create_label(example):
    return {"labels": [label2id.get(label, -1) for label in example["condition"]]}

tokenized_datasets = tokenized_datasets.map(create_label, batched=True)
print(tokenized_datasets)

def filter_label(example):
    return example["labels"] != -1

tokenized_datasets = tokenized_datasets.filter(filter_label)
print(tokenized_datasets)




print(tokenized_datasets["train"]["condition"][:5])
sample_labels = tokenized_datasets["train"]["labels"][:5]
for label in sample_labels:
    print(id2label[label])

tokenized_datasets = tokenized_datasets.remove_columns(['condition'])
tokenized_datasets.set_format("torch")

print(tokenized_datasets["train"][0])


#dynamic padding
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator)


for step, batch in enumerate(train_dataloader):
    print(batch["input_ids"].shape)
    if step > 5:
        break

# Training the model
from transformers import AutoModelForSequenceClassification, AdamW, get_scheduler
checkpoint = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=len(condition_list))

optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = 1
num_training_steps = num_epochs * len(train_dataloader)

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)
model.to(device)

from tqdm.auto import tqdm
progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for index, batch in enumerate(train_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

        if index % 2 == 0: 
            print(f"\nEpoch [{epoch}/{num_epochs}], Step [{index}/{len(train_dataloader)}], loss: {loss.item():.4f}")

        if index == 10:
            break

'''
saving the model on huggingface hub
'''

# from huggingface_hub import login
# access_token_write = "abc" #read from the website
# login(token = access_token_write)

# model.push_to_hub("dummy")
# tokenizer.push_to_hub("dummy")  #
# model.config.id2label = id2label
# model.config.label2id = label2id       

# Evaluating the model
from datasets import load_metric
from transformers import AutoTokenizer
import pandas as pd
import pandas as pd
metric = load_metric("accuracy")

eval_dataloader = DataLoader(tokenized_datasets['validation'], batch_size=8, collate_fn=data_collator)

from tqdm.auto import tqdm

progress_bar = tqdm(range(len(eval_dataloader)))

model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])
    progress_bar.update(1)

metric.compute()
