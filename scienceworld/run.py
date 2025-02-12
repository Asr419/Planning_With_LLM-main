import os
import json
import argparse

from hotpotqa import HotPotQATask
from models_custom import gpt_usage
from lats import lats_search
# from hotpot_qa.mcts_sc import lats_sc_search
from tot import dfs_search
from rap import mcts_search
import logging
import random

def run(args):
    task = HotPotQATask()
    logs, cnt_avg, cnt_any = [], 0, 0

    # create log directories if they don't exist
    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    
    
    logging.basicConfig(filename=args.log, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filemode='a')

    count = 0
    task_accs = []
    f1_score=[]
    info = []
    idxs = list(range(100))
    random.Random(230).shuffle(idxs)

    for i in range(args.task_start_index, args.task_end_index):
    # for i in idxs[:100]:
        # solve
        if args.algorithm == 'lats':
            state, value, all_nodes, reward, em, f1 = lats_search(args, task, i, args.iterations, True)
        elif args.algorithm == 'tot':
            state, value, all_nodes, reward, em, f1 = dfs_search(args, task, i, args.iterations)
        elif args.algorithm == 'rap':
            state, value, all_nodes, reward, em, f1 = mcts_search(args, task, i, args.iterations)
        else:
            raise Exception("Search algorithm option not valid")
         # log main metric
        if em is None:
            em = 0
        task_accs.append(em)
        print(f"F1 score : {f1}")
        if f1 is None:
            f1=0
        f1_score.append(f1)
        cnt_avg = sum(task_accs) / len(task_accs)
        logging.info(f"Result EM : {sum(task_accs) / len(task_accs)}")
        logging.info(f"F1 score : {sum(f1_score) / len(f1_score)}")

        print(i, 'len(task_accs)', len(task_accs), 'cnt_avg', cnt_avg, '\n')
        #all_nodes_dict = [(node.to_dict(), value) for node, value in all_nodes]
        
       
    n = args.task_end_index - args.task_start_index
    print('usage_so_far', gpt_usage(args.backend))

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--backend', type=str, choices=['gpt-4', 'gpt-35-turbo'], default='gpt-35-turbo')
    args.add_argument('--temperature', type=float, default=1.0)
    args.add_argument('--task_start_index', type=int, default=900)
    args.add_argument('--task_end_index', type=int, default=1000)
    args.add_argument('--prompt_sample', type=str, choices=['standard', 'cot'])
    args.add_argument('--n_generate_sample', type=int, default=1)  
    args.add_argument('--n_evaluate_sample', type=int, default=1)
    args.add_argument('--iterations', type=int, default=50)
    args.add_argument('--log', type=str)
    args.add_argument('--algorithm', type=str, choices=['lats', 'rap','tot'], default = ['lats'])

    args = args.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)
    run(args)