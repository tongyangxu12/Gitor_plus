import argparse
import os
import time
import networkx as nx
from nodevectors import ProNE
import pprint
from concurrent.futures import ProcessPoolExecutor, as_completed
import extract
import metrics
from datetime import datetime


# executor = ProcessPoolExecutor(max_workers=11)
# task_list = [
#     executor.submit(side_info.add_nodes, graph1, args.source_func_path),
#     executor.submit(side_info.add_nodes, graph2, args.source_func_path)
# ]
# process_results = [task.result() for task in as_completed(task_list)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embed_dim', type=int, default=16)
    parser.add_argument('--source_func_path', default='./test/')
    # parser.add_argument('--metrics_choice', nargs='+', default= ['getxmet', 'getvref', 'getvdec', 'getnos', 'getnopr', 'getnoa',
    #                                                              'getnexp', 'getnand', 'getlmet', 'getexct', 'getexcr', 'getcref', 'getcomp',
    #                                                              'getcast', 'getnbltrl', 'getncltrl', 'getnsltrl', 'getnnltrl', 'getnnulltrl',
    #                                                              'gethvoc', 'gethdif', 'getheff', 'getmdn', 'getloop', 'getmnp', 'getnfci', 'getndi'])
    # parser.add_argument('--metrics_name')
    args = parser.parse_args()

    print('INFO: --------args----------')
    for k in list(vars(args).keys()):
        print('INFO: %s: %s' % (k, vars(args)[k]))
    print('INFO: --------args----------\n')

    current_time = datetime.now().strftime("%H:%M")
    print("Now is:", current_time)
    print('\n')

    begin_time = time.time()

    G1 = nx.DiGraph()
    G2 = nx.DiGraph()

    graph1 = extract.add_nodes(G1, './data/id2sourcecode')
    # graph1 = extract.add_nodes(G1, './data/filtered_code_files')

    graph2 = metrics.add_nodes(G2, './data/id2sourcecode')
    # graph2 = metrics.add_nodes(G2, './data/filtered_code_files')
    graph = nx.compose(graph1, graph2)

    g2v_1 = ProNE(
        n_components=args.embed_dim
    )
    # g2v_2 = ProNE(
    #     n_components=args.embed_dim
    # )
    # g2v_3 = ProNE(
    #     n_components=args.embed_dim
    # )
    # g2v_4 = ProNE(
    #     n_components=args.embed_dim
    # )
    # print(graph3)

    # executor = ProcessPoolExecutor(
    #     max_workers=11)
    # task_list = [
    #     executor.submit(extract.add_nodes, G1, './data/id2sourcecode'),
    #     executor.submit(metrics.add_nodes, G3, './data/id2sourcecode')
    # ]
    # process_results = [task.result() for task in as_completed(task_list)]


    g2v_1.fit(graph)
    # g2v_2.fit(graph1)
    # g2v_3.fit(graph2)
    # g2v_4.fit(graph3)

    print(f'\n\nINFO: all time is {time.time() - begin_time}s')

    g2v_1.save(os.path.join(f'./test1/dim_{args.embed_dim}', "embedding"))
    print("node list:", graph.nodes())
    print("edge list:", graph.edges())

    # print(graph)
    # print(graph1)
    # print(graph2)

    # print(f"Graph1: Nodes = {graph1.number_of_nodes()}, Edges = {graph1.number_of_edges()}")
    # print(f"Graph2: Nodes = {graph2.number_of_nodes()}, Edges = {graph2.number_of_edges()}")
    print(f"Graph: Nodes = {graph.number_of_nodes()}, Edges = {graph.number_of_edges()}")

    # for edge in G1.edges(data=True):
    #     print(f"Edge {edge[0]} -> {edge[1]}, Weight: {edge[2].get('weight', 1)}")

    # graph2 = extract.add_nodes()
    # pp = pprint.PrettyPrinter(indent=4)
    # pp.pprint(dict(graph1.adj))

