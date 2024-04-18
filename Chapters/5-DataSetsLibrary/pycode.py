from datasets import load_dataset

data_files = {"train": "Data/drugsComTrain_raw.tsv", "test": "Data/drugsComTest_raw.tsv"}
# \t is the tab character in Python
drug_dataset = load_dataset("csv", data_files=data_files, delimiter="\t")
drug_sample = drug_dataset["train"].shuffle(seed=42).select(range(1000))
# print(drug_sample[0]) range(1000) = [0, 1, 2, ..., 999]

drug_dataset = drug_dataset.rename_column(original_column_name="Unnamed: 0", 
                                          new_column_name="patient_id")
print(drug_dataset)
# print(drug_sample[:2])

for split in drug_dataset.keys():
    assert len(drug_dataset[split]) == len(drug_dataset[split].unique("patient_id"))

drug_dataset = drug_dataset.filter(lambda x: x["condition"] is not None)

def lowercase_condition(example):
    return {"condition": example["condition"].lower()}

# modifying current column
drug_dataset = drug_dataset.map(lowercase_condition)

print(drug_dataset["train"]["condition"][:3])

# creating a new column
def compute_review_length(example):
    return {"review_length": len(example["review"].split())}

drug_dataset = drug_dataset.map(compute_review_length) 
print(drug_dataset)

# print(drug_dataset["train"].sort("review_length")[:2])
print(drug_dataset.num_rows)

import html

text = "I&#039;m a transformer called BERT"
print(html.unescape(text))

#cleaning the review column from html character codes
drug_dataset = drug_dataset.map(lambda x: {"review": html.unescape(x["review"])})

# map code super power
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


def tokenize_function(examples):
    return tokenizer(examples["review"], truncation=True)

tokenized_dataset = drug_dataset.map(tokenize_function, batched=True)
print("============")
print(drug_dataset)
print(tokenized_dataset)