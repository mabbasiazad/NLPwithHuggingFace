from datasets import load_dataset
#==============================================================================
data_files = {"train": "Data/drugsComTrain_raw.tsv", "test": "Data/drugsComTest_raw.tsv"}
# \t is the tab character in Python
drug_dataset = load_dataset("csv", data_files=data_files, delimiter="\t")

# printing the dataset and its first row
drug_sample = drug_dataset["train"].shuffle(seed=42).select(range(1000))
print(drug_sample[0]) #range(1000) = [0, 1, 2, ..., 999]

# =============================================================================
# renaming the first column
drug_dataset = drug_dataset.rename_column(original_column_name="Unnamed: 0", 
                                          new_column_name="patient_id")
print(drug_dataset)

# =============================================================================
# filtering non "condition" values
drug_dataset = drug_dataset.filter(lambda x: x["condition"] is not None)

def lowercase_condition(example):
    return {"condition": example["condition"].lower()}

drug_dataset = drug_dataset.map(lowercase_condition)
drug_dataset = drug_dataset.filter(lambda x: "</span>" not in x["condition"])
# Check that lowercasing worked
print(drug_dataset["train"]["condition"][:3])


# =============================================================================
# creating a new column: review_length
def compute_review_length(example):
    return {"review_length": len(example["review"].split())}

drug_dataset = drug_dataset.map(compute_review_length)
print(drug_dataset)

# filtering the dataset based on review length
drug_dataset = drug_dataset.filter(lambda x: x["review_length"] > 30)
print(drug_dataset.num_rows)

# =============================================================================
drug_dataset_clean = drug_dataset["train"].train_test_split(train_size=0.8, seed=42)
# Rename the default "test" split to "validation"
drug_dataset_clean["validation"] = drug_dataset_clean.pop("test")
# Add the "test" set to our `DatasetDict`
drug_dataset_clean["test"] = drug_dataset["test"]
print(drug_dataset_clean)

drug_dataset_clean.save_to_disk("Projects/ConditionPrediction/drug-reviews-data")

