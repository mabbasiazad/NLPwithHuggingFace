from datasets import load_dataset
#==============================================================================
# the path calculated relative to the path on cmd line when you run your program
data_files = {"train": "Data/drugsComTrain_raw.tsv", "test": "Data/drugsComTest_raw.tsv"}
# \t is the tab character in Python
drug_dataset = load_dataset("csv", data_files=data_files, delimiter="\t")
print(drug_dataset)

import psutil

# Process.memory_info is expressed in bytes, so convert to megabytes
print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

train_dataset = drug_dataset["train"]
print(train_dataset)
print(f"Number of files in dataset : {train_dataset.dataset_size}")
size_gb = train_dataset.dataset_size / (1024**3)
print(f"Dataset size (cache file) : {size_gb:.2f} GB")
