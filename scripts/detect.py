import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import random
import torch
import torch.nn as nn
from nodevectors import ProNE
import sklearn
import gensim
print(sklearn.__version__)


ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED_DIR = ROOT / 'data' / 'processed'
GENERATED_DIR = ROOT / 'results' / 'generated'

def write_results_to_file(output_file, content):
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(content + '\n')

word2index = {}
index2word = {}
func_num = 0

file_list = os.listdir(DATA_PROCESSED_DIR / 'id2sourcecode')
for file in file_list:
    tmp_file = int(file.split('.')[0])

    if tmp_file not in word2index:
        word2index[tmp_file] = func_num + 1
        index2word[func_num + 1] = tmp_file
        func_num += 1
    else:
        raise ValueError

w = ProNE.load(str(GENERATED_DIR / 'testboth' / 'dim_16' / 'embedding.zip'))
w1 = ProNE.load(str(GENERATED_DIR / 'testkeyword' / 'dim_16' / 'embedding.zip'))
w2 = ProNE.load(str(GENERATED_DIR / 'testsideinfo' / 'dim_16' / 'embedding.zip'))
w3 = gensim.models.KeyedVectors.load_word2vec_format(str(GENERATED_DIR / 'exp1' / 'code2vec_emb.txt'), binary=False)
w4 = ProNE.load(str(GENERATED_DIR / 'testmetrics' / 'dim_16' / 'embedding.zip'))

wtype = ['both', 'keyword', 'sideinfo', 'metrics']
dim = ['16', '32', '64', '128']

weights = torch.randn(func_num + 1, 16)
weights[0] = torch.zeros(16)

weights1 = torch.randn(func_num + 1, 16)
weights1[0] = torch.zeros(16)

weights2 = torch.randn(func_num + 1, 16)
weights2[0] = torch.zeros(16)

weights4 = torch.randn(func_num + 1, 16)
weights4[0] = torch.zeros(16)

funcs_cnt = 0

for word, func_idx in word2index.items():
    word = f'{word}.java'
    try:
        weights[func_idx] = torch.FloatTensor(w.predict(word))
        weights1[func_idx] = torch.FloatTensor(w1.predict(word))
        weights2[func_idx] = torch.FloatTensor(w2.predict(word))
        weights4[func_idx] = torch.FloatTensor(w4.predict(word))
        funcs_cnt += 1
    except:
        print(f'ERROR: {word}')
        pass

def get_csv(path, sample_size=250000):
    eval_dataset = []
    df = pd.read_csv(path, sep=',', engine='python')
    for _, row in df.iterrows():
        f1 = int(row['f1'])
        f2 = int(row['f2'])
        eval_dataset.append([f1, f2])
    if len(eval_dataset) > sample_size:
        eval_dataset = random.sample(eval_dataset, sample_size)
    return eval_dataset

def get_csv1(path):
    eval_dataset = []
    df = pd.read_csv(path, sep=',', engine='python')
    for _, row in df.iterrows():
        f1 = int(row['f1'])
        f2 = int(row['f2'])
        eval_dataset.append([f1, f2])
    return eval_dataset

def cal_sim(f1, f2, manhattan_dis=None):
    f1, f2 = word2index[f1], word2index[f2]
    wf1, wf2 = weights4[f1], weights4[f2]
    return torch.cosine_similarity(wf1, wf2, dim=0)

csv_dir = DATA_PROCESSED_DIR / 'BCB'
if 'csv' in str(csv_dir):
    csv_list = [str(csv_dir)]
else:
    csv_list = [str(csv_dir / i) for i in os.listdir(csv_dir)]

manhattan_dis = nn.PairwiseDistance(p=1)
tp = tn = fp = fn = 0

result_list = []

for csv_file in csv_list:
    print(f'\n\nINFO: Loading {csv_file}...')

    eval_dataset = get_csv(csv_file)

    print(f'INFO: The length of {csv_file} is {len(eval_dataset)}')
    result_list.append(f'INFO: The length of {csv_file} is {len(eval_dataset)}')

    tmp_tp = tmp_fn = 0

    for i in eval_dataset:
        f1, f2 = i[0], i[1]
        tmp_sim = cal_sim(f1, f2, manhattan_dis=manhattan_dis)

        if 'NoClone' not in csv_file:
            if tmp_sim >= 0.7:
                tp += 1
                tmp_tp += 1
            else:
                fn += 1
                tmp_fn += 1
        elif 'NoClone' in csv_file:
            if tmp_sim < 0.7:
                tn += 1
            else:
                fp += 1

    if 'NoClone' not in csv_file:
        print(f'INFO: *csv: {csv_file}, tp: {tmp_tp}, fn: {tmp_fn}, R: {float(tmp_tp / (tmp_tp + tmp_fn))} ')
        result_list.append(
            f'INFO: csv: {csv_file}, tp: {tmp_tp}, fn: {tmp_fn}, R: {float(tmp_tp / (tmp_tp + tmp_fn))} ')

print(f'\n\nINFO: #####################final result#####################')
print(f'*INFO: tp: {tp}, tn: {tn}, fp: {fp}, fn: {fn}')
P = float(tp / (tp + fp))
R = float(tp / (tp + fn))
f1 = float(2.0 * P * R / (P + R))
print(f'*INFO: P: {P}, R: {R}, f1: {f1}')
result_list.append(f'\n\nINFO: #####################final result#####################')
result_list.append(f'INFO: tp: {tp}, tn: {tn}, fp: {fp}, fn: {fn}')
result_list.append(f'*INFO: P: {P}, R: {R}, f1: {f1}')

timestamp = datetime.now().strftime('%m-%d_%H-%M')
result_dir = GENERATED_DIR / 'result'
result_dir.mkdir(parents=True, exist_ok=True)
output_file = os.path.join(str(result_dir), f'results_{timestamp}_{wtype[3]}_{dim[0]}.txt')

for result in result_list:
    write_results_to_file(output_file, result)
