import pandas as pd
import random


# Read all CSV files and combine them into one complete data set
csv_files = [
    './data/BCB/Clone-pairs-MT3.csv',
    './data/BCB/Clone-pairs-ST3.csv',
    './data/BCB/Clone-pairs-T1.csv',
    './data/BCB/Clone-pairs-T2.csv',
    './data/BCB/Clone-pairs-T4.csv',
    './data/BCB/Clone-pairs-VST3.csv',
    './data/BCB/NoClone-pairs.csv'
]
# Stores all cloned and non-cloned pairs
all_data = []

# Read all csv files and merge
for csv_file in csv_files:
    df = pd.read_csv(csv_file, names=['f1', 'f2', 'type'], header=None)

    # Convert the type column to 0 or 1, where 1 represents a cloned pair and 0 represents a non-cloned pair
    df['type'] = df['type'].apply(lambda x: 1 if pd.notnull(x) else 0)

    all_data.extend(df.values.tolist())

# Total data
total_pairs = len(all_data)
print(f"Total data：{total_pairs} pairs")

# The target total data volume is 1 million pairs
total_selected_pairs = 1000000
# Ensure that the total data selected does not exceed the total data
total_selected_pairs = min(total_selected_pairs, total_pairs)

# 1 million pairs were randomly selected from all the data
random.shuffle(all_data)
selected_data = all_data[:total_selected_pairs]

# Calculate the size of the training, test, and verification sets
# train_size = int(0.8 * total_selected_pairs)
# test_size = val_size = int(0.1 * total_selected_pairs)

# Partition dataset
# train_data = selected_data[:train_size]
# test_data = selected_data[train_size:train_size + test_size]
# val_data = selected_data[train_size + test_size:]


# Save as csv file (training set, test set, verification set)
def save_data_to_csv(data, file_path):
    df = pd.DataFrame(data, columns=['f1', 'f2', 'type'])
    df.to_csv(file_path, index=False)


# save_data_to_csv(train_data, './data/split_data/train.csv')
# save_data_to_csv(test_data, './data/split_data/test.csv')
# save_data_to_csv(val_data, './data/split_data/val.csv')
save_data_to_csv(selected_data, './data/split_data/all.csv')

# print(f"Train set size：{len(train_data)}")
# print(f"Test set size：{len(test_data)}")
# print(f"Verification set size：{len(val_data)}")
print(f"size：{len(selected_data)}")