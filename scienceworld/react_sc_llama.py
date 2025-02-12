import os, time, json, logging, argparse, random, re, sys
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from scienceworld import ScienceWorldEnv

load_dotenv()

log_folder = "hotpot_qa/logs_sc/react/llama3-8b"
if not os.path.exists(log_folder):
    os.makedirs(log_folder)

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Configure episodic results logs
log_file3 = os.path.join(log_folder, "results_data_10.log")
logger3 = logging.getLogger('logger3')
logger3.setLevel(logging.INFO)
file_handler3 = logging.FileHandler(log_file3, mode='a')
formatter3 = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler3.setFormatter(formatter3)
logger3.addHandler(file_handler3)
logger3.propagate = False

class LLaMA2LM:
    def __init__(self, model_name="meta-llama/Meta-Llama-3-70B"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map = "auto")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, max_new_tokens=600)

    def __call__(self, prompt, max_length=4096, stop=None, **kwargs):
        response = self.pipe(prompt, max_length=max_length, num_return_sequences=1, eos_token_id=self.tokenizer.eos_token_id)
        
        generated_text = response[0]['generated_text']
        
        
        if stop:
            stop_pos = generated_text.find(stop[0])
            if stop_pos != -1:
                generated_text = generated_text[:stop_pos]
        # print(f"Generating : {generated_text}")
        
         
        return generated_text

class ScienceWorld_ReAct():
    def __init__(self):
        self.lm = LLaMA2LM()
        self.prompt_i = 0
        folder = 'hotpot_qa/prompts/'
        prompt_file = 'scienceworld_react.json'
        with open(folder + prompt_file, 'r') as f:
            prompt_dict = json.load(f)
        self.webthink_examples = prompt_dict['webthink_simple']
        self.instruction = """Solve a science task by answering with interleaving Thought, Action steps. You will get the observation from the environment so don't need to generate it. Thought can reason about the current situation, and Action has to be one out of the valid actions provided. Make sure to always generate an Action.
        """
        self.exitCommands = ["quit", "exit"]

    def webthink(self, taskIdx, numEpisodes, simplificationStr, args, logger1, to_print=True):
        print(f"Step Limit: {args['env_step_limit']}")
        env = ScienceWorldEnv("", args['jar_path'], envStepLimit=args['env_step_limit'])
        taskNames = env.get_task_names()
        taskName = taskNames[taskIdx]
        maxVariations = env.get_max_variations(taskName)
        finalScores = []
        env.load(taskName, 0, "")
        maxVariations = env.get_max_variations(taskName)
        time.sleep(2)

        for episodeIdx in range(0, numEpisodes):
            randVariationIdx = env.get_random_variation_test()
            env.load(taskName, randVariationIdx, simplificationStr)
            templates, lut = env.get_possible_action_object_combinations()
            templates_list = [d['action'] for d in templates]

            logger1.info(f"Object IDX to Object Referent LUT: " + str(lut))
            logger1.info(f"Task Name: " + taskName)
            logger1.info("Task Variation: " + str(randVariationIdx) + " / " + str(maxVariations))
            logger1.info("Task Description: " + str(env.get_task_description()))
            logger1.info("look: " + str(env.look()))
            logger1.info("inventory: " + str(env.inventory()))
            logger1.info("taskdescription: " + str(env.taskdescription()))

            self.webthink_prompt = self.instruction + self.webthink_examples + f"TaskDescription : {env.taskdescription()}"
            initial_prompt = f"Generate a thought to solve the task. TaskDescription : {env.taskdescription()}"
            initial_thought = self.lm(initial_prompt, stop=None)
            print(f"Initial Thought : {initial_thought}")
            self.webthink_prompt += f"\nThought:{initial_thought}"

            score = 0.0
            isCompleted = False
            curIter = 0
            userInputStr = "look around"  # First action

            while (userInputStr not in self.exitCommands) and (isCompleted is False):
                logger1.info("----------------------------------------------------------------")
                logger1.info("Step: " + str(curIter))
                logger1.info("----------------------------------------------------------------")

                observation, reward, isCompleted, info = env.step(userInputStr)
                poss_actions = env.get_possible_actions()
                poss_objects = env.get_possible_objects()

                if curIter > 30:
                    isCompleted = True
                score = info['score']
                self.webthink_prompt += f"\nAction: {userInputStr}\nObservation: {observation}"

                logger1.info("Observation: \n>>> " + observation)
                logger1.info("Reward: " + str(reward))
                logger1.info("Score: " + str(score))
                logger1.info("isCompleted: " + str(isCompleted))

                if isCompleted:
                    break

                validActions = env.get_valid_action_object_combinations_with_templates()
                action_list = [d['action'] for d in validActions]
                gpt_action_str = self.lm(self.webthink_prompt + f"\nValid Actions : {action_list}\n", stop=[f"\nObservation:"])
                print(f"Output by model : {gpt_action_str}")

                # print(f"Output by LLaMA2: {gpt_action_str}")
                pattern = r"Action:\s*(.*)"
                match = re.search(pattern, gpt_action_str, re.IGNORECASE)

                if match:
                    action = match.group(1).strip()
                else:
                    gpt_action_str = self.lm(self.webthink_prompt + f"\nAction Templates : {templates_list}\n", stop=[f"\nObservation:"])
                    match = re.search(pattern, gpt_action_str, re.IGNORECASE)
                    if match:
                        action = match.group(1).strip()
                        logger1.info(f"Template action given : {action}")
                    else:
                        gpt_action_str = self.lm(self.webthink_prompt + f"\nGenerate a Thought and an Action where the action should be based on the possible actions and possible observations.\nPossible type of actions : {poss_actions} and OBJ can be taken from possible type of objects : {poss_objects}\n", stop=[f"\nObservation:"])
                        logger1.info(f"Generated LLAMACHAT Output : {gpt_action_str}")
                        # print(f"Output by LLaMA2: {gpt_action_str}")
                        match = re.search(pattern, gpt_action_str, re.IGNORECASE)
                        if match:
                            action = match.group(1).strip()
                            logger1.info(f"LLaMA2 action taken : {action}")
                        else:
                            action = "look around"

                userInputStr = str(action).lower().strip()
                # logger1.info("Output by LLaMA2: " + gpt_action_str)
                # logger1.info("Action given by LLaMA2: " + str(userInputStr))
                # print("Action given by LLaMA2: " + str(userInputStr))
                curIter += 1

            logger1.info("Goal Progress:")
            logger1.info(env.get_goal_progress())
            time.sleep(1)

            finalScores.append(score)
            logger1.info("Final score: " + str(score))
            logger1.info("isCompleted: " + str(isCompleted))

            filenameOutPrefix = args['output_path_prefix'] + str(taskIdx)
            env.store_run_history(episodeIdx, notes={'text': 'my notes here'})
            env.save_run_histories_buffer_if_full(filenameOutPrefix, max_per_file=args['max_episode_per_file'])

        env.save_run_histories_buffer_if_full(filenameOutPrefix, max_per_file=args['max_episode_per_file'], force_save=True)

        avg = sum([x for x in finalScores if x >= 0]) / len(finalScores)
        logger3.info("")
        logger3.info("---------------------------------------------------------------------")
        logger3.info(" Summary (React Agent)")
        logger3.info(" Task " + str(taskIdx) + ": " + taskName)
        logger3.info(" Simplifications: " + str(simplificationStr))
        logger3.info("---------------------------------------------------------------------")
        logger3.info(" Episode scores: " + str(finalScores))
        logger3.info(" Average episode score: " + str(avg))
        logger3.info("---------------------------------------------------------------------")
        logger3.info("")

        logger1.info("Completed.")
        sys.stdout.flush()
        self.prompt_i += 1
        return 0, 0

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

    simplification_group = parser.add_argument_group('Game simplifications')
    simplification_group.add_argument("--simplifications-preset", choices=['easy'],
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

def main():
    logging.info("ScienceWorld 1.0 API Examples - Random Agent")
    # Parse command line arguments
    idxs=list(range(30))

    args = parse_args()
    random.seed(args["seed"])
    args["simplification_str"] = build_simplification_str(args)
    simplificationStr = args['simplification_str']
    numEpisodes=args['num_episodes']
    
    
    for i in idxs[0:30]:
        log_file1 = os.path.join(log_folder, f"react_data_task_{i}.log")
        logger1 = logging.getLogger('logger1')
        logger1.setLevel(logging.INFO)
        file_handler1 = logging.FileHandler(log_file1, mode='a')
        formatter1 = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler1.setFormatter(formatter1)
        logger1.addHandler(file_handler1)
        logger1.propagate = False
        Sc_ReAct=ScienceWorld_ReAct()
        
        r, info = Sc_ReAct.webthink(i, numEpisodes,simplificationStr, args, logger1,to_print=True)
        file_handler1.close()
        logger1.removeHandler(file_handler1)

if __name__=="__main__":
    main()