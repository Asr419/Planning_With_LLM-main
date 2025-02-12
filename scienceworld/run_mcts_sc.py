import os
import json
import argparse

import logging
import random
from scworld_task import ScWorldTask
from mcts_sc_rollout import mcts_sc_search
from mcts_llama_sc_rollout import mcts_llama_sc_search
import random
from scienceworld import ScienceWorldEnv


log_folder = "hotpot_qa/logs_sc/lats_seed_try"
if not os.path.exists(log_folder):
    os.makedirs(log_folder)

log_folder1 = "hotpot_qa/logs_sc/lats_seed_try"
if not os.path.exists(log_folder1):
    os.makedirs(log_folder1)


log_file3 = os.path.join(log_folder, "result_test_30.log")
logger3 = logging.getLogger('logger3')
logger3.setLevel(logging.INFO)
file_handler3 = logging.FileHandler(log_file3, mode='a')
formatter3 = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler3.setFormatter(formatter3)
logger3.addHandler(file_handler3)
logger3.propagate = False

def build_simplification_str(args):
    """ Build simplification_str from args. """
    simplifications = list()
    if args["teleport"]:
        simplifications.append("teleportAction")

    if args["self_watering_plants"]:
        simplifications.append("selfWateringFlowerPots")

    if args["open_containers"]:
        simplifications.append("openContainers")

    if args["open_doors"]:
        simplifications.append("openDoors")

    if args["no_electrical"]:
        simplifications.append("noElectricalAction")

    return args["simplifications_preset"] or ",".join(simplifications)

def parse_args():
    desc = "Run a model that chooses random actions until successfully reaching the goal."
    parser = argparse.ArgumentParser(desc)
    parser.add_argument("--jar_path", type=str,
                        help="Path to the ScienceWorld jar file. Default: use builtin.")
    parser.add_argument("--task-num", type=int, default=13,
                        help="Specify the task number to play. Default: %(default)s")
    parser.add_argument("--var-num", type=int, default=0,
                        help="Specify the task variation number to play. Default: %(default)s")
    parser.add_argument("--env-step-limit", type=int, default=20,
                        help="Maximum number of steps per episode. Default: %(default)s")
    parser.add_argument("--num-episodes", type=int, default=5,
                        help="Number of episodes to play. Default: %(default)s")
    parser.add_argument("--seed", type=int, default=1001,
                        help="Seed the random generator used for sampling random actions.")
    parser.add_argument("--output-path-prefix", default="save-histories",
                        help="Path prefix to use for saving episode transcripts. Default: %(default)s")
    parser.add_argument("--max-episode-per-file", type=int, default=1000,
                        help="Maximum number of episodes per transcript file. Default: %(default)s")
    parser.add_argument("--n_generate_sample", type=int, default= 3, help="Number of sampled actions which is interleaving of thought and action.")
    parser.add_argument('--n_evaluate_sample', type=int, default=1)

    simplification_group = parser.add_argument_group('Game simplifications')
    simplification_group.add_argument("--simplifications-preset", choices=['easy'], default='easy',
                                      help="Choose a preset among: 'easy' (apply all possible simplifications).")
    simplification_group.add_argument("--teleport", action="store_true",
                                      help="Lets agents instantly move to any location.")
    simplification_group.add_argument("--self-watering-plants", action="store_true",
                                      help="Plants do not have to be frequently watered.")
    simplification_group.add_argument("--open-containers", action="store_true",
                                      help="All containers are opened by default.")
    simplification_group.add_argument("--open-doors", action="store_true",
                                      help="All doors are opened by default.")
    simplification_group.add_argument("--no-electrical", action="store_true",
                                      help="Remove the electrical actions (reduces the size of the action space).")

    args = parser.parse_args() 
    params = vars(args)
    return params

def main():
    logging.info("ScienceWorld 1.0 API Examples - Random Agent")
    # Parse command line arguments
    taskidxs=list(range(30))
    
    args = parse_args()
    args["simplification_str"] = build_simplification_str(args)
    simplificationStr = args['simplification_str']
    numEpisodes=args['num_episodes']
    env=ScienceWorldEnv("", args['jar_path'], envStepLimit=args['env_step_limit'])
    task=ScWorldTask()
    seeds=[2,3,5,7,9]
    for j in taskidxs[27:30]:
        
        log_file2 = os.path.join(log_folder, f"Planning_task_{j}_v1_t1.log")
        logger2 = logging.getLogger('logger2')
        logger2.setLevel(logging.INFO)
        file_handler2 = logging.FileHandler(log_file2, mode='a')
        formatter2 = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler2.setFormatter(formatter2)
        logger2.addHandler(file_handler2)
        logger2.propagate = False
        taskNames=env.get_task_names()
        taskName=taskNames[j]
        env.load(taskName, 0, "")
        maxVariations = env.get_max_variations(taskName)
        final_score=[]
        for episodeIdx in range(0, numEpisodes) : 
            # random.seed(args["seed"])
            random.seed(seeds[episodeIdx])
 
            randVariationIdx = env.get_random_variation_test()
            env.reset()
            #call mcts_sc_search for llm scoring and mcts_llama_sc_search for fintetuned llama
            state,value,all_nodes,reward, score=mcts_sc_search(args,env,task,randVariationIdx,taskName,simplificationStr,logger2)
            logger3.info(" Task " + str(j) + ": " + taskName)
            logger3.info("Task Variation: " + str(randVariationIdx) + " / " + str(maxVariations))
            logger3.info("Task Description: " + str(env.get_task_description()))
            logger3.info(f"Task NAme :" + taskName)
            logger3.info("Reward: "+str(reward))
            logger3.info("Final score: "+str(score))
            final_score.append(score)
        logger3.info(f"All episode score : {final_score}")
        final_score = [0 if x == -100 else x for x in final_score]
        avg_score=sum(final_score)/len(final_score)
        logger3.info(f"Average episodic score{avg_score}")
        file_handler2.close()
        logger2.removeHandler(file_handler2)
if __name__=="__main__":
    main()