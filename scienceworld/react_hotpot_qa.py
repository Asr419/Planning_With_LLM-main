import requests
import os
from dotenv import load_dotenv
import wikienv as wikienv, wrappers as wrappers
import json
import sys
import random
import time
import datetime
from dsp import LM
import sys
from dotenv import load_dotenv
import logging


log_folder = "hotpot_qa/logs/react"
if not os.path.exists(log_folder):
    os.makedirs(log_folder)

# Set up logging
# log_file = os.path.join(log_folder, f"react_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
log_file = os.path.join(log_folder, f"react_trial.log")
logging.basicConfig(filename=log_file, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s', filemode= 'a')


CALL_COUNT_FILE = "call_counts.txt"
load_dotenv()
def step(env, action):
    attempts = 0
    while attempts < 10:
        try:
            return env.step(action)
        except requests.exceptions.Timeout:
            attempts += 1


def get_openai_response(azure_url, ims_org_id, api_key, temp_auth_token, json_data):
    headers = {
        'x-gw-ims-org-id': ims_org_id,
        'x-api-key': api_key,
        'Authorization': f'Bearer {temp_auth_token}',
        'Content-Type': 'application/json',
    }
    response = requests.post(azure_url, headers=headers, json=json_data)
    return json.loads(response.text)

def read_call_count():
    try:
        with open(CALL_COUNT_FILE, "r") as file:
            return int(file.read())
    except FileNotFoundError:
        return 0

def write_call_count(count):
    with open(CALL_COUNT_FILE, "w") as file:
        file.write(str(count))

# def read_call_count():
#     try:
#         with open(CALL_COUNT_FILE, "r") as file:
#             # Read all lines from the file
#             lines = file.readlines()
#             # Initialize count as 0
#             count = 0
#             # Loop through lines to find the count for today's date
#             for line in lines:
#                 # Split the line into date and count
#                 parts = line.strip().split(",")
#                 # Check if the date matches today's date
#                 if parts[0] == datetime.date.today().strftime("%Y-%m-%d"):
#                     count = int(parts[1])
#                     break
#             return count
#     except FileNotFoundError:
#         return 0

# def write_call_count(count):
#     # Open the file in append mode to add the count for today's date
#     with open(CALL_COUNT_FILE, "a") as file:
#         # Get today's date in the required format
#         today_date = datetime.date.today().strftime("%Y-%m-%d")
#         # Write today's date and count as a line in the file
#         file.write(f"{today_date},{count}\n")


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


    def basic_request(self, prompt,stop=None, **kwargs):
        # Prompt = f"There is a 5x5 maze and you are at (0,0). You need to reach the goal at (4,4). "
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
                            "model_name": "gpt-35-turbo",
                            "temperature": 0,
                            "max_tokens": 100,
                            "top_p": 1,
                            "frequency_penalty": 0,
                            "presence_penalty": 0,
                            "llm_type": "azure_chat_openai",
                            "stop":stop
                        },
                        
                    }
        # if self.json_data['llm_metadata']['content_filter'] != "safe":
        #     jo
        max_retries = 4
        retry_count = 0
        # openai_response = get_openai_response(self.azure_url, self.ims_org_id, self.api_key, self.temp_auth_token, self.json_data)
        while retry_count < max_retries:
            try:
                openai_response = get_openai_response(self.azure_url, self.ims_org_id, self.api_key, self.temp_auth_token, self.json_data)
                if 'generations' not in openai_response:
                    raise KeyError("Generations key not found in response")
                break  # Exit loop if response is successful
            except KeyError as e:
                print(f"KeyError: {e}. Retrying...")
                retry_count += 1
        else:
            print("Max retries exceeded. Failed to get response.")
        # print(openai_response)
        if 'generations' not in openai_response:
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

    def __call__(self, prompt,stop=None, only_completed=True, return_sorted=False, **kwargs):
        sys.stdout.flush()
        return self.basic_request(prompt,stop,**kwargs)


class Hotpot_React():
    def __init__(self):
        self.lm=FireLM()
        folder = 'hotpot_qa/prompts/'
        prompt_file = 'prompts_naive.json'
        with open(folder + prompt_file, 'r') as f:
            prompt_dict = json.load(f)
      
        

        self.webthink_examples = prompt_dict['webthink_simple6']
        
        self.instruction = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
        (1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
        (2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
        (3) Finish[answer], which returns the answer and finishes the task.
        Here are some examples.
        """
        self.webthink_prompt = self.instruction + self.webthink_examples
    def run_gpt3(self, prompt, engine='text-davinci-003', max_tokens=200,stop=None):
        call_count = read_call_count()
        call_count += 1
        write_call_count(call_count)
        sys.stdout.flush()
        return self.lm(prompt, stop)

    def webthink(self,idx=None, to_print=True):
        question = env.reset(idx=idx)
        if to_print:
            logging.info(f"idx : {idx}, question : {question}")
        self.webthink_prompt += question + "\n"
        n_calls, n_badcalls = 0, 0
        for i in range(1, 8):
            n_calls += 1
            thought_action = self.run_gpt3(self.webthink_prompt + f"Thought {i}:", stop=[f"\nObservation {i}:"])
            try:
                thought, action = thought_action.strip().split(f"\nAction {i}: ")
            except:
                logging.info(f"Thought_action : {thought_action}")
                n_badcalls += 1
                n_calls += 1
                thought = thought_action.strip().split('\n')[0]
                action = self.run_gpt3(self.webthink_prompt + f"Thought {i}: {thought}\nAction {i}:", stop=[f"\n"]).strip()
            if len(action)>=2:
                obs, r, done, info = step(env, action[0].lower() + action[1:])
                obs = obs.replace('\\n', '')
            else: 
                done = False
            step_str = f"Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {obs}\n"
            self.webthink_prompt += step_str
            if to_print:
                logging.info(f"Prompt : {step_str}")
            if done:
                break
        if not done:
            obs, r, done, info = step(env, "finish[]")
        if to_print:
            logging.info(f"Info : {info}")
        info.update({'n_calls': n_calls, 'n_badcalls': n_badcalls, 'traj': self.webthink_prompt})
        sys.stdout.flush()
        return r, info

if __name__=="__main__":
    logging.info("Hotpot_React initialized.")
    load_dotenv()
    client_secret=os.getenv("CLIENT_SECRET")
    auth_key=os.getenv("AUTH_KEY") 
    lm=FireLM()
    env = wikienv.WikiEnv()
    env = wrappers.HotPotQAWrapper(env, split="train")
    env = wrappers.LoggingWrapper(env)
    idxs = list(range(7405))
    # random.Random(233).shuffle(idxs)
    
    rs = []
    infos = []
    f1=[]
    old_time = time.time()
    for i in idxs[:100]:
        hotpot_react=Hotpot_React()
        r, info = hotpot_react.webthink(i, to_print=True)
        rs.append(info['em'])
        f1.append(info['f1'])
        infos.append(info)
        # print(info)
        # print(rs)
        print(sum(rs))
        print(len(rs))
        print(sum(rs), len(rs), sum(rs) / len(rs), (time.time() - old_time) / len(rs))
        logging.info(f"Result EM : {sum(rs)/len(rs)}")
        logging.info(f"F1 score : {sum(f1)/len(rs)}")
        print('-----------')
        print()
        print(i)
    print("its done")