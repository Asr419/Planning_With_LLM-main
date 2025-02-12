#Environment change to scienceworld
import os, time, json, requests, sys
import random
import logging
import argparse
import ast

from dotenv import load_dotenv
from dsp import LM
import re
from scienceworld import ScienceWorldEnv

load_dotenv()
log_folder = "hotpot_qa/logs_sc/seed_try"
if not os.path.exists(log_folder):
    os.makedirs(log_folder)

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Configure episode logs
log_file1 = os.path.join(log_folder, "react_trial_template.log")
logger1 = logging.getLogger('logger1')
logger1.setLevel(logging.INFO)
file_handler1 = logging.FileHandler(log_file1, mode='a')
formatter1 = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler1.setFormatter(formatter1)
logger1.addHandler(file_handler1)
logger1.propagate = False

# Configure valid action logs
log_file2 = os.path.join(log_folder, "sc_validations_template_actions.log")
logger2 = logging.getLogger('logger2')
logger2.setLevel(logging.INFO)
file_handler2 = logging.FileHandler(log_file2, mode='a')
formatter2 = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler2.setFormatter(formatter2)
logger2.addHandler(file_handler2)
logger2.propagate = False

# Configure epsiodic results logs
log_file3 = os.path.join(log_folder, "results1.log")
logger3 = logging.getLogger('logger3')
logger3.setLevel(logging.INFO)
file_handler3 = logging.FileHandler(log_file3, mode='a')
formatter3 = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler3.setFormatter(formatter3)
logger3.addHandler(file_handler3)
logger3.propagate = False

def convert_to_dict(s):
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError) as e:
        print(f"Error converting string to dictionary: {e}")
        return None
    
def get_openai_response(azure_url, ims_org_id, api_key, temp_auth_token, json_data):
    headers = {
        'x-gw-ims-org-id': ims_org_id,
        'x-api-key': api_key,
        'Authorization': f'Bearer {temp_auth_token}',
        'Content-Type': 'application/json',
    }
    response = requests.post(azure_url, headers=headers, json=json_data)
    return json.loads(response.text)

class FireLM(LM):
    def __init__(self,**kwargs):
        self.client_secret=os.getenv("CLIENT_SECRET") 
        self.auth_key=os.getenv("AUTH_KEY")
     
        self.kwargs=kwargs
        self.provider = "default"
        self.history = []

        
        # GPT-3.5 setup
        self.CLIENT_ID='MDSR_Firefall'
        self.CLIENT_SECRET=self.client_secret
        self.AUTH_KEY=self.auth_key 
        self.IMS_URL='https://ims-na1-stg1.adobelogin.com/ims/token/v2'
        self.AZURE_CHAT_COMPLETION_START= 'https://firefall-stage.adobe.io/v1/chat/completions/conversations'
        self.AZURE_CHAT_COMPLETION= 'https://firefall-stage.adobe.io/v1/chat/completions'
        self.AZURE_COMPLETIONS='https://firefall-stage.adobe.io/v1/completions'
        self.azure_url = 'https://firefall-stage.adobe.io/v1/completions'
        self.ims_org_id = self.CLIENT_ID
        self.api_key = self.CLIENT_ID

     

        self.params = {
            'client_id': self.CLIENT_ID,
            'client_secret': self.CLIENT_SECRET,
            'code': self.AUTH_KEY,
            'grant_type': 'authorization_code',
        }
        self.response = requests.post(self.IMS_URL, data=self.params)

        self.temp_auth_token=json.loads(self.response.text)['access_token']


    def basic_request(self, prompt,model="gpt-4",stop=None, **kwargs):
        # Prompt = f"There is a 5x5 maze and you are at (0,0). You need to reach the goal at (4,4). 
        data = {
            **kwargs,
            # "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        self.json_data =  {
                        "dialogue":{
                            "question": prompt
                        },
                        "llm_metadata": {
                            "model_name": model,
                            "temperature": 0.5,
                            "max_tokens": 100,
                            "top_p": 1,
                            "frequency_penalty": 0,
                            "presence_penalty": 0,
                            "llm_type": "azure_chat_openai",
                            "stop":stop
                        },
                        
                    }
        # if self.json_data['llm_metadata']['content_filter'] != "safe":
        max_retries = 4
        retry_count = 0
        openai_response = get_openai_response(self.azure_url, self.ims_org_id, self.api_key, self.temp_auth_token, self.json_data)
   

        # while retry_count < max_retries:
        #     try:
        #         openai_response = get_openai_response(self.azure_url, self.ims_org_id, self.api_key, self.temp_auth_token, self.json_data)
        #         if 'generations' not in openai_response:
        #             raise KeyError("Generations key not found in response")
        #         break  # Exit loop if response is successful
        #     except KeyError as e:
        #         logging.info(f"KeyError: {e}. Retrying...")
        #         retry_count += 1
        # else:
        #     logging.info("Max retries exceeded. Failed to get response.")
        # logging.info(openai_response)
        if 'generations' not in openai_response:
            print(openai_response)
            severity_prompt = "content filter"
            return severity_prompt
        
        openai_response['choices']=openai_response['generations'][0]  # to adhere to dspy nomencleture
        
        self.history.append({
            "prompt": prompt,
            "response": openai_response,
            "kwargs": kwargs,
        })

        # text = [generated_text['text'] for generated_text in openai_response['generations'][0]]
        text=openai_response['choices'][0]["text"]
        sys.stdout.flush()
        return text

    def __call__(self, prompt,model="gpt-4",stop=None, only_completed=True, return_sorted=False, **kwargs):
        sys.stdout.flush()
        return self.basic_request(prompt,model,stop,**kwargs)
    
class ScienceWorld_ReAct():
    def __init__(self):
        self.lm= FireLM()
        self.prompt_i=0
        folder = 'hotpot_qa/prompts/'
        prompt_file = 'scienceworld_react.json'
        with open(folder + prompt_file, 'r') as f:
            prompt_dict = json.load(f)
        self.webthink_examples = prompt_dict['webthink_simple']
        # self.webthink_examples = prompt_dict[f'Task_{self.prompt_i}']
        self.instruction = """Solve a science task by answering with interleaving Thought, Action steps. You will get the observation from the environment so don't need to generate it. Thought can reason about the current situation, and make sure to generate Action which has to be one out of the valid actions provided.
        """
        self.exitCommands = ["quit","exit"]
    def run_gpt3(self, prompt, model="gpt-4",engine='text-davinci-003', max_tokens=200,stop=None):
            sys.stdout.flush()
            return self.lm(prompt,model,stop)
    
    def webthink(self, taskIdx, numEpisodes,simplificationStr, args, seed,to_print=True):
        print(f"Step Limit : {args['env_step_limit']}")
        env = ScienceWorldEnv("", args['jar_path'], envStepLimit=args['env_step_limit'])
        exitCommands = ["quit", "exit"]
        taskNames=env.get_task_names()
        taskName=taskNames[taskIdx]
        maxVariations = env.get_max_variations(taskName)
        finalScores=[]
        env.load(taskName, 0, "")
        maxVariations = env.get_max_variations(taskName)
        time.sleep(2)
        for episodeIdx in range(0, numEpisodes) : 
            random.seed(seed[episodeIdx])
            
            randVariationIdx = env.get_random_variation_train()
            env.reset()
            env.load(taskName, randVariationIdx, simplificationStr)
            templates, lut = env.get_possible_action_object_combinations()
            templates_list = [d['action'] for d in  templates]
            
            # print(f"templates : {len(templates_list)}")
            # print(f"templates_list :{templates_list}")
            logger1.info(f"Object IDX to Object Referent LUT: " + str(lut))
            logger1.info(f"Task NAme :" + taskName)
            logger1.info("Task Variation: " + str(randVariationIdx) + " / " + str(maxVariations))
            logger1.info("Task Description: " + str(env.get_task_description()))
            logger1.info("look: " + str(env.look()))
            logger1.info("inventory: " + str(env.inventory()))
            logger1.info("taskdescription: " + str(env.taskdescription()))
            self.webthink_prompt=self.instruction+self.webthink_examples+f"TaskDescription : {env.taskdescription()}"
            initial_prompt=f"Generate a thought to solve the task. TaskDescription : {env.taskdescription()}"
            initial_thought=self.run_gpt3(initial_prompt, stop=None)
            self.webthink_prompt+=f"\nThought:{initial_thought}"

            score = 0.0
            isCompleted = False
            curIter = 0
            userInputStr = "look around"        # First action
            while (userInputStr not in exitCommands) and (isCompleted is False):
                logger1.info("----------------------------------------------------------------")
                logger1.info("Step: " + str(curIter))
                logger1.info("----------------------------------------------------------------")
                logger2.info("Step: " + str(curIter))
            
                observation, reward, isCompleted, info = env.step(userInputStr)
                poss_actions=env.get_possible_actions()
                poss_objects=env.get_possible_objects()
                # print(f"No. of actions : {len(poss_actions)}")
                # print(f"No. of objects : {len(poss_objects)}")

                # print(f"get possible actions: {len(env.get_possible_actions())}")
                # print(f"Possible actions : {env.get_possible_actions()}")
                # print(f"get_poosible objects: {len(env.get_possible_objects())}")
                # print(f"Possible objects : {env.get_possible_objects()}")
                if curIter > 40:
                    isCompleted=True
                score=info['score']
                self.webthink_prompt+=f"\nAction: {userInputStr}\nObservation: {observation}"

                logger1.info("Observation: \n>>> " + observation)
                logger1.info("Reward: " + str(reward))
                logger1.info("Score: " + str(score))
                logger1.info("isCompleted: " + str(isCompleted))
                in_context=f"Task"

                if (isCompleted):
                    break
                validActions = env.get_valid_action_object_combinations_with_templates()
                #gpt-validaActions
                action_list = [d['action'] for d in validActions]
                

                
                gpt_action_str=self.run_gpt3(self.webthink_prompt + f"\nPossible type of actions : {poss_actions} and OBJ can be taken from possible type of objects ; {poss_objects}\n Your response should always return a Thought and an Action\n ",model="gpt-4", stop=[f"\nObservation:"])
                print(f"Output by gpt4: {gpt_action_str}")
                pattern = r"Action:\s*(.*)"
                match = re.search(pattern, gpt_action_str, re.IGNORECASE)
                action = match.group(1).strip()
                logger1.info(f"GPT-4 action taken : {action}") 
                      
              
                userInputStr = str(action)
                userInputStr = userInputStr.lower().strip()
                logger1.info("Output by GPT: " + gpt_action_str)
                logger1.info("Action given by GPT: " + str(userInputStr))
                print("Action given by GPT: " + str(userInputStr))
                curIter += 1
                

            logger1.info("Goal Progress:")
            logger1.info(env.get_goal_progress())
            time.sleep(1)

            # Episode finished -- Record the final score
            finalScores.append(score)
            logger1.info("Final score: " + str(score))
            logger1.info("isCompleted: " + str(isCompleted))

            filenameOutPrefix = args['output_path_prefix'] + str(taskIdx)
            env.store_run_history(episodeIdx, notes={'text': 'my notes here'})
            env.save_run_histories_buffer_if_full(filenameOutPrefix, max_per_file=args['max_episode_per_file'])
         # Episodes are finished -- manually save any last histories still in the buffer
        env.save_run_histories_buffer_if_full(filenameOutPrefix, max_per_file=args['max_episode_per_file'], force_save=True)

        # Clip negative scores to 0 for average calculation
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
        self.prompt_i+=1
        return 0,0      
        

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


#
#   Parse command line arguments
#
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
    parser.add_argument("--seed", type=int,
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

def main():
    logging.info("ScienceWorld 1.0 API Examples - Random Agent")
    # Parse command line arguments
    idxs=list(range(30))

    args = parse_args()
    seeds=[2,3,5,7,9]
    random.seed(args["seed"])
    args["simplification_str"] = build_simplification_str(args)
    simplificationStr = args['simplification_str']
    numEpisodes=args['num_episodes']
    Sc_ReAct=ScienceWorld_ReAct()
    
    for i in idxs[29:30]:
        
        r, info = Sc_ReAct.webthink(i, numEpisodes,simplificationStr, args, seed=seeds, to_print=True)

if __name__=="__main__":
    main()