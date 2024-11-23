import os
import argparse
from datetime import datetime
from nodevectors import ProNE
import torch
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from itertools import combinations


def process_file_batch(args):
    # Processing the similarity calculation of a batch of files
    w, file_pairs, threshold = args
    clones = []
    for file1, file2 in file_pairs:
        wf1 = torch.FloatTensor(w.predict(file1))
        wf2 = torch.FloatTensor(w.predict(file2))
        cosine = torch.cosine_similarity(wf1, wf2, dim=0)
        if cosine > threshold:
            clones.append((file1, file2))
    return clones


def compare_files_in_directory(directory, w, args):
    # Get all the '.java' files
    files = [f for f in os.listdir(directory) if f.endswith('.java')]

    # Split the file pair into multiple batches, processing only one batch at a time
    batch_size = len(files) // args.n_workers + 1  # Number of file pairs per batch
    batches = []

    # Generate batches of file pairs one by one and pass them one by one to parallel tasks
    for i in range(0, len(files), batch_size):
        # Use a generator to avoid generating a large number of file pairs at once
        batch = combinations(files[i:i + batch_size], 2)
        batches.append(batch)

    begin_time = time.time()

    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
        futures = [
            executor.submit(process_file_batch, (w, batch, args.cosine))
            for batch in batches
        ]

        all_clones = []
        for future in as_completed(futures):
            all_clones.extend(future.result())

    test_time = time.time() - begin_time

    print(f"Found {len(all_clones)} clone pairs")

    print(f"test_time: {test_time}s")

    with open(f'./scalability_embed/dim_{args.embed_dim}/test_time.txt', 'a+') as ff:
        ff.write(f'gaijin_test_time: {test_time}s    num:{args.num}\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embed_dim', type=int, default=16)
    parser.add_argument('--cosine', type=float, default=0.7)
    parser.add_argument('--n_workers', type=int, default=11)  # By default, 11 worker processes are used
    parser.add_argument('--num', type=int, default=1)
    args = parser.parse_args()

    print('INFO: --------args----------')
    for k, v in vars(args).items():
        print(f'INFO: {k}: {v}')
    print('INFO: --------args----------\n')

    current_time = datetime.now().strftime("%H:%M")
    print("Now is:", current_time)

    # Loading model
    w = ProNE.load(f'./scalability_embed/dim_{args.embed_dim}/embedding_{args.num}.zip')
    directory = f'./data/filtered_code_files'

    # GPU
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        print("Using CUDA for computations")

    compare_files_in_directory(directory, w, args)