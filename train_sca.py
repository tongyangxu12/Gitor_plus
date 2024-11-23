import argparse
import os
import time
import networkx as nx
from nodevectors import ProNE
import extract
import metrics
from datetime import datetime
from multiprocessing import Pool
import numpy as np


def create_combined_graph(loc_path):
    """Create combined graph using parallel processing"""
    G1 = nx.DiGraph()
    G2 = nx.DiGraph()
    graph1 = extract.add_nodes(G1, loc_path)
    graph2 = metrics.add_nodes(G2, loc_path)
    return nx.compose(graph1, graph2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embed_dim', type=int, default=16)
    parser.add_argument('--embed_method', choices=['prone', 'fast'], default='prone')
    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--num', type=int, default=1)
    args = parser.parse_args()

    print('INFO: --------args----------')
    for k, v in vars(args).items():
        print(f'INFO: {k}: {v}')
    print('INFO: --------args----------\n')

    current_time = datetime.now().strftime("%H:%M")
    print("Now is:", current_time)

    # Create graph
    loc_path = f'../data/filtered_code_files'

    # Configure ProNE for faster processing
    g2v_1 = ProNE(
        n_components=args.embed_dim,
        step=5,  # Adjust step size for faster convergence
        mu=0.2,  # Adjust regularization parameter
        theta=0.5,  # Adjust propagation parameter
    )

    begin_time = time.time()
    graph = create_combined_graph(loc_path)


    # Train the model
    g2v_1.fit(graph)

    train_time = time.time() - begin_time
    print(f"train_time: {train_time}s")

    # Save results
    output_dir = f'./scalability_embed/dim_{args.embed_dim}'
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'train_time.txt'), 'a+') as ff:
        ff.write(f'gaijin_train_time: {train_time}s    num: {args.num}\n')

    g2v_1.save(os.path.join(output_dir, f"embedding_{args.num}"))