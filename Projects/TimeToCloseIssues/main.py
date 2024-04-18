"""
Problem Description:
Try it out! Calculate the average time it takes to close issues in 🤗 Datasets. 
You may find the Dataset.filter() function useful to filter out the pull requests 
and open issues (state is open), and you can use the Dataset.set_format() function to convert the datase
t to a DataFrame so you can easily manipulate the created_at and closed_at timestamps. 
"""

from datasets import load_dataset

issues_dataset = load_dataset("lewtun/github-issues", split="train")
print(issues_dataset)

#filter dataset to get just issues (not pull requests) which are closed
closed_issues_dataset = issues_dataset.filter(lambda x: x["is_pull_request"] == False)
closed_issues_dataset = closed_issues_dataset.filter(lambda x: x["state"] == "closed")

print(closed_issues_dataset[0])

#convert dataset to DataFrame
closed_issues_dataset.set_format("pandas")
# print(closed_issues_dataset[0:3])

df = closed_issues_dataset[:]

import pandas as pd
# pandas.set_option('display.max_columns', None)
# pandas.set_option('display.width', 1000)

print(df.head())

# Convert 'created_at' and 'closed_at' columns to datetime
df['created_at'] = pd.to_datetime(df['created_at'])
df['closed_at'] = pd.to_datetime(df['closed_at'])
print(df['created_at'])

# Calculate time taken to close each issue
df['time_to_close'] = df['closed_at'] - df['created_at']

# Calculate average time taken to close issues
average_time_to_close = df['time_to_close'].mean()

print("Average time to close issues:", average_time_to_close)



closed_issues_dataset.reset_format()

