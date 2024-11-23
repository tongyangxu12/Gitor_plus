import os
import pandas as pd
import torch
import random
import shutil
from nodevectors import ProNE

def save_concatenated_vectors_1(csv_file, output_dir, w):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = pd.read_csv(f'./data/BCB/{csv_file}')
    if len(df) > 250000:
        df = df.sample(n=250000, random_state=42)

    for index, row in df.iterrows():
        file1 = row['f1']
        file2 = row['f2']
        vector1 = w.predict(f'{file1}.java')  # Gets the vector of the first file
        vector2 = w.predict(f'{file2}.java')  # Gets the vector of the second file

        # Concatenation vector
        concatenated_vector = torch.cat((torch.tensor(vector1, dtype=torch.float32),
                                         torch.tensor(vector2, dtype=torch.float32)), dim=0)

        # Save to file
        # output_file = os.path.join(output_dir, f'{file1}_{file2}.pt')  # Create the output file name using the file name
        output_file = os.path.join(output_dir, f'{file1}_{file2}.pt')
        torch.save(concatenated_vector, output_file)  # Save as .pt file


def save_concatenated_vectors_0(csv_file, output_dir, w):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = pd.read_csv(f'./data/BCB/{csv_file}')
    if len(df) > 250000:
        df = df.sample(n=250000, random_state=42)

    for index, row in df.iterrows():
        file1 = row['f1']
        file2 = row['f2']
        vector1 = w.predict(f'{file1}.java')  # Gets the vector of the first file
        vector2 = w.predict(f'{file2}.java')  # Gets the vector of the second file

        # 拼接向量
        concatenated_vector = torch.cat((torch.tensor(vector1, dtype=torch.float32),
                                         torch.tensor(vector2, dtype=torch.float32)), dim=0)

        # 保存到文件
        # output_file = os.path.join(output_dir, f'{file1}_{file2}.pt')  # Create the output file name using the file name
        output_file = os.path.join(output_dir, f'{file1}_{file2}_0_.pt')
        torch.save(concatenated_vector, output_file)  # Save as .pt file

def split_and_save(vectors_dir, output_dir, vectors_dirs, seed=42, train_ratio=0.8, val_ratio=0.1):
    # Set random seeds
    random.seed(seed)

    # Get all.pt files
    files = [f for f in os.listdir(vectors_dir) if f.endswith('.pt')]
    random.shuffle(files)  # Randomly shuffles the file order

    total_files = len(files)
    train_size = int(total_files * train_ratio)
    val_size = int(total_files * val_ratio)

    # Partition file
    train_files = files[:train_size]
    val_files = files[train_size:train_size + val_size]
    test_files = files[train_size + val_size:]

    train_dir = os.path.join(vectors_dirs, output_dir[0])
    val_dir = os.path.join(vectors_dirs, output_dir[1])
    test_dir = os.path.join(vectors_dirs, output_dir[2])

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    # Save to a different folder
    for file_name in train_files:
        shutil.copy(os.path.join(vectors_dir, file_name), train_dir)

    for file_name in val_files:
        shutil.copy(os.path.join(vectors_dir, file_name), val_dir)

    for file_name in test_files:
        shutil.copy(os.path.join(vectors_dir, file_name), test_dir)

    print(f"Data set partition complete：\nTraining set: {len(train_files)}\nValidation set: {len(val_files)}\nTest set: {len(test_files)}")

def getvet(w, output_dir1, output_dir2, output_dir3):

    # for i in ['Clone-pairs-T1.csv', 'Clone-pairs-T2.csv', 'Clone-pairs-VST3.csv', 'Clone-pairs-ST3.csv',
    #           'Clone-pairs-MT3.csv', 'Clone-pairs-T4.csv']:
    #     save_concatenated_vectors_1(i, output_dir1, w)
    split_and_save(output_dir1, ['trainvec', 'valvec', 'testvec'], output_dir3)

    # save_concatenated_vectors_0('NoClone-pairs.csv', output_dir2, w)
    split_and_save(output_dir2, ['trainvec', 'valvec', 'testvec'], output_dir3)


if __name__ == '__main__':
    # Generate data sets for training, validation, and testing
    # w1 = ProNE.load('./metrics_keyword/dim_16/embedding.zip')
    # w2 = ProNE.load('./metrics_keyword/dim_32/embedding.zip')
    # w3 = ProNE.load('./metrics_keyword/dim_64/embedding.zip')
    # w4 = ProNE.load('./metrics_keyword/dim_128/embedding.zip')
    #
    # output_dir1 = ['../vec/16', '../vec/32', '../vec/64', '../vec/128']
    # output_dir2 = ['../vec_0/16', '../vec_0/32', '../vec_0/64', '../vec_0/128']
    # output_dir3 = ['./datavec/dim_16', './datavec/dim_32', './datavec/dim_64', './datavec/dim_128']
    #
    # getvet(w1, output_dir1[0], output_dir2[0], output_dir3[0])
    # getvet(w2, output_dir1[1], output_dir2[1], output_dir3[1])
    # getvet(w3, output_dir1[2], output_dir2[2], output_dir3[2])
    # getvet(w4, output_dir1[3], output_dir2[3], output_dir3[3])

    # scalability
    w = ProNE.load('./scalability_embed/dim_16/1000000/embedding_1.zip')
    output_dir = './scalability_embed/dim_16/1000000/fcl_sca'


