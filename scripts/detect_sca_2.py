import os
import torch
import time
import concurrent.futures
from pathlib import Path
from nodevectors import ProNE


ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED_DIR = ROOT / 'data' / 'processed'
GENERATED_DIR = ROOT / 'results' / 'generated'


def get_embeddings(directory, w):
    embeddings = {}
    files = [f for f in os.listdir(directory) if f.endswith('.java')]
    for file in files:
        embeddings[file] = torch.FloatTensor(w.predict(file))
    return embeddings


def is_clone(wf1, wf2, threshold):
    cosine = torch.cosine_similarity(wf1, wf2, dim=0)
    return cosine > threshold


def compare_file_pair(file_pair, embeddings, args):
    file1, file2 = file_pair
    wf1, wf2 = embeddings[file1], embeddings[file2]
    if is_clone(wf1, wf2, args.cosine):
        print(f'Clone detected between: {file1} and {file2}')


def compare_files_in_directory(directory, w, args):
    embeddings = get_embeddings(directory, w)
    files = list(embeddings.keys())
    file_pairs = [(files[i], files[j]) for i in range(len(files)) for j in range(i + 1, len(files))]

    begin_time = time.time()
    for pair in file_pairs:
        compare_file_pair(pair, embeddings, args)
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     futures = [executor.submit(compare_file_pair, pair, embeddings, args) for pair in file_pairs]
    #     concurrent.futures.wait(futures)

    total = time.time() - begin_time
    print(f'test_time: {total:.2f}s')
    output_dir = GENERATED_DIR / 'scalability_embed' / f'dim_{args.embed_dim}'
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'test_time.txt', 'a+') as ff:
        ff.write(f'1 test_time: {total}s    num:{args.num}\n')
        ff.close()


if __name__ == '__main__':
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser()
    parser.add_argument('--embed_dim', type=int, default=16)
    parser.add_argument('--cosine', type=float, default=0.7)
    parser.add_argument('--num', type=int, default=1)
    args = parser.parse_args()

    w = ProNE.load(str(GENERATED_DIR / 'scalability_embed' / f'dim_{args.embed_dim}' / f'embedding_{args.num}.zip'))
    directory = str(DATA_PROCESSED_DIR / 'filtered_code_files')
    compare_files_in_directory(directory, w, args)
