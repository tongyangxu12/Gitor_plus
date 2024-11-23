import os
import pandas as pd
import shutil


csv_path = './data/split_data/all.csv'
data = pd.read_csv(csv_path)


file_names = set(data['f1'].tolist() + data['f2'].tolist())


source_folder = './data/id2sourcecode'
target_folder = './data/filtered_code_files'

os.makedirs(target_folder, exist_ok=True)

for file_name in file_names:
    source_path = os.path.join(source_folder, f"{file_name}.java")
    target_path = os.path.join(target_folder, f"{file_name}.java")
    if os.path.exists(source_path):
        shutil.copy(source_path, target_path)
    else:
        print(f"file {file_name}.java does not exist.")

print("done!")