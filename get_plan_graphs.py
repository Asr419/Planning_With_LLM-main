# %%
# 
# 1. Adapting FireFly to dspy format
# 2. Generating multiple trajectories for task 5 times, plotting them to get a directional tree

# %%
from dsp import LM
import requests
import json
import os
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
        self.kwargs=kwargs
        self.provider = "default"
        self.history = []
        
        # GPT-3.5 setup
        self.CLIENT_ID='MDSR_Firefall'
        self.CLIENT_SECRET=''
        self.AUTH_KEY='' #imss -> service tokens -> permanent auth token
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


    def basic_request(self, prompt, **kwargs):
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
                            "temperature": 0.2,
                            "max_tokens": 1000,
                            "top_p": 1,
                            "frequency_penalty": 0,
                            "presence_penalty": 0,
                            "n": 5,
                            "llm_type": "azure_chat_openai",
                        },
                        
                    }

        openai_response = get_openai_response(self.azure_url, self.ims_org_id, self.api_key, self.temp_auth_token, self.json_data)
        openai_response['choices']=openai_response['generations'][0]  # to adhere to dspy nomencleture
        print(openai_response)
        self.history.append({
            "prompt": prompt,
            "response": openai_response,
            "kwargs": kwargs,
        })

        text = [generated_text['text'] for generated_text in openai_response['generations'][0]]
        return text

    def __call__(self, prompt, only_completed=True, return_sorted=False, **kwargs):
        return self.basic_request(prompt,**kwargs)



# %%
# GPT-3 HLP generator

import pandas as pd
# import openai
import random
from ast import literal_eval
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

import re
import networkx as nx
import matplotlib.pyplot as plt
def plot_trajectories(trajectories,task,start='',end=''):
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add edges for each trajectory
    for trajectory in trajectories:
        edges = [(trajectory[i], trajectory[i + 1]) for i in range(len(trajectory) - 1)]
        G.add_edges_from(edges)
    
    # Draw the graph
    pos = nx.spring_layout(G)  # Use spring layout for better visualization
    plt.figure(figsize=(10, 7))
    nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightblue', font_size=7, font_weight='bold', arrowsize=20)
    plt.title(f'task: {task[0]},  start: {start},   end:{end}')
    plt.show()


def remove_numbering(plan_list):
    # Initialize an empty list to store the modified plans
    modified_list = []
    
    # Define a regex pattern to match numbering at the start of a string
    pattern = re.compile(r'^\d+\.\s*')
    
    # Iterate through each element in the list
    for plan in plan_list:
        # Remove the numbering using the regex pattern
        modified_plan = pattern.sub('', plan)
        # Add the modified plan to the list
        modified_list.append(modified_plan)
    
    # Remove empty strings from the list
    modified_list = [plan for plan in modified_list if plan]
    
    return modified_list

def split_by_plans(plan_list):
    # Remove numbering from the list
    plan_list = remove_numbering(plan_list)
    
    # Initialize lists for each plan range
    plan1_to_plan2 = []
    plan2_to_plan3 = []
    plan3_to_plan4 = []
    plan4_to_plan5 = []
    plan5_onwards = []

    # Initialize the current range
    current_range = plan1_to_plan2
    
    # Populate the lists based on the plan ranges
    for plan in plan_list:
        if 'Plan 1:' in plan:
            current_range = plan1_to_plan2
        elif 'Plan 2:' in plan:
            current_range = plan2_to_plan3
        elif 'Plan 3:' in plan:
            current_range = plan3_to_plan4
        elif 'Plan 4:' in plan:
            current_range = plan4_to_plan5
        elif 'Plan 5:' in plan:
            current_range = plan5_onwards
        
        current_range.append(plan)
    
    return plan1_to_plan2[1:], plan2_to_plan3[1:], plan3_to_plan4[1:], plan4_to_plan5[1:], plan5_onwards[1:]


def get_plan(p):
    p=p.split('\n')  
    p=remove_numbering(p)
    p=split_by_plans(p)
    return p



ACT_TO_STR = {
    'OpenObject': "Open",
    'CloseObject': "Close",
    'PickupObject': "Pickup",
    'PutObject': "Put",
    'ToggleObjectOn': "Toggle on",
    'ToggleObjectOff': "Toggle off",
    'SliceObject': "Slice",
    'Navigation': "Navigate"
}


class LLM_HLP_Generator():
    def __init__(self, knn_data_path, emb_model_name='paraphrase-MiniLM-L6-v2', debug=False):
        self.sentence_embedder = SentenceTransformer(emb_model_name)
        from transformers import GPT2Tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.knn_set = pd.read_pickle(knn_data_path)
        self.debug=debug
        self.lm=FireLM()


    def knn_retrieval(self, curr_task, k):
        # Find K train examples with closest sentence embeddings to test example
        
        traj_emb = self.sentence_embedder.encode(curr_task["task_instr"])
        topK = []
        for idxTrain, trainItem in self.knn_set.iterrows():

            train_emb = self.sentence_embedder.encode(trainItem["task_instr"])

            dist = -1 * cos_sim(traj_emb, train_emb)

            if len(topK) < k:
                topK.append((trainItem["task"], dist))
                topK = sorted(topK, key = lambda x : x[1])
            else:
                if float(dist) < topK[-1][1]:
                    if (trainItem["task"], dist) not in topK:
                        topK.append((trainItem["task"], dist))
                        topK = sorted(topK, key = lambda x : x[1])
                        topK = topK[:k]

        return [entry[0] for entry in topK]


    def generate_prompt(self, curr_task, k, removeNav=False, naturalFormat=False, includeLow=False):
        #header
        prompt = "Create a high-level plan for completing a household task using the allowed actions and visible objects."
        
        if naturalFormat:
            prompt += f"\n\n\nAllowed actions: {', '.join(ACT_TO_STR.values())}" 
        else:
            prompt += f"\n\n\nAllowed actions: {', '.join(ACT_TO_STR.keys())}" 
        
        #prompt += "Valid objects in the environment: f{}"

        # Run KNN retrieval
        knn_retrieved_examples = self.knn_retrieval(curr_task, k)

        # Add in-context examples from knn retrieval
        for retrieved_task in knn_retrieved_examples:
            trainTaskRow = self.knn_set.loc[self.knn_set["task"] == retrieved_task]
            trainTaskRow = trainTaskRow.iloc[0]


            step_list = [literal_eval(listItem) for rowItem in trainTaskRow["gold_traj"] for listItem in rowItem]

            #REMOVE NAVIGATION STEPS if the flag is set
            if removeNav:
                stepListCleaned = []
                for listItem in step_list:
                    if "Navigation" not in listItem:
                        stepListCleaned.append(listItem)
                step_list = stepListCleaned
            
            # Format action names to be more natural
            if naturalFormat:
                stepListCleaned = []
                for listItem in step_list:
                    listItem = list(listItem)
                    act_str = ACT_TO_STR[listItem[0]] 
                    listItem[0] = act_str
                    stepListCleaned.append(tuple(listItem))
                step_list = stepListCleaned
            
            # Split past and next plans randomly
            planSplit = random.sample(range(len(step_list)),1)[0]
            
            # In-context examples components
            high_level_str = str(trainTaskRow["task_instr"])
            step_by_step_str = '. '.join(trainTaskRow["step_instr"])
            past_plan_str = self.format_plan_str(step_list[:planSplit])
            next_plans_str = self.format_plan_str(step_list[planSplit:])
            in_context_obj_str = self.format_object_str(trainTaskRow["vis_objs"])

            # In-context examples
            prompt += "\n\nTask description: " + high_level_str \
                    
            # Include low-level instructions
            if includeLow:
                prompt += "\nStep by step instructions: " + step_by_step_str

            prompt +=  "\nCompleted plans: " + past_plan_str  \
                    + "\nVisible objects are " + in_context_obj_str \
                    + "\nNext Plans: " + next_plans_str
                    
        
        # Add the task prompt for GPT-3
        ## In-context examples components
        completed_plans = curr_task["completed_plans"]
        vis_objs = curr_task["vis_objs"]

        task_high_level_str = str(curr_task["task_instr"][0])
        task_step_by_step_str = '. '.join(curr_task["step_instr"])
        task_past_plan_str = self.format_plan_str(completed_plans)
        task_obj_str = self.format_object_str(vis_objs)

        # Example for above strings
        # task_high_level_str = 'Cook the potato and put it into the recycle bin.'
        # task_step_by_step_str = '. '.join(curr_task["step_instr"])
        # task_past_plan_str = self.format_plan_str(completed_plans)
        # task_obj_str = 'microwave, fridge, potato, garbagecan'

        prompt += "\n\nTask description: " + task_high_level_str \
                

        if includeLow:
            prompt += "\nStep by step instructions: " + task_step_by_step_str

        prompt += "\nCompleted plans: " + task_past_plan_str \
                + "\nVisible objects are " + task_obj_str \
                + "\nGenerate 5 distinct plans achieving the task having same start and end step."
                # + "\nNext Plans:"
                
        
        curr_task["Prompts"] = prompt
        curr_task["vis_objs"] = vis_objs
        print(prompt)
        return prompt


    #run GPT-3 on specified test set using the KNN prompts
    def run_gpt3(self, prompt, logit_bias_text, engine='text-davinci-003', max_tokens=200):
        return self.lm(prompt)
         

    # Main point of entry for LLM HLP generator
    def generate_hlp(self, curr_task, k):

        prompt = self.generate_prompt(curr_task, k, removeNav=False, naturalFormat=False, includeLow=False)

        generated_hlp = self.run_gpt3(prompt, curr_task["vis_objs"])

        return generated_hlp 

    # Main point of entry for LLM HLP prompt generator, use in run_eval
    def generate_gpt_prompt(self, curr_task, k):

        prompt = self.generate_prompt(curr_task, k, removeNav=False, naturalFormat=False, includeLow=False)
        # print(prompt)
        return prompt 

    
    # Below are helper functions 

    # Change object list into object string:
    ## Example: ['Drawer', 'ButterKnife'] -> Drawer, ButterKnife
    def format_object_str(self, obj_list):

        if not obj_list:
            return ""

        obj_str = ", ".join(obj_list).lower()
        return obj_str

    # Change plan list into plan string:
    ## Example: [('Navigation','Shelf'), ('PickupObject', 'knife')] -> Navigation Shelf. PickupObject Knife
    def format_plan_str(self, plan_list):

        if not plan_list:
            return ""

        # Lowercase object names in (action, plan) tuple
        lowercased_plan_list = []
        for item in plan_list:
            item_list = list(item)
            item_list[1] = item_list[1].lower()
            if len(item_list) > 2:
                item_list[2] = item_list[2].lower()
            lowercased_plan_list.append(tuple(item_list))

        plans = [" ".join(item) for item in lowercased_plan_list]
        plan_str = ', '.join(plans)

        return plan_str


import numpy as np
    
if __name__=='__main__':
    curr_task = {
                    "task_instr": ["Cook the potato and put it into the recycle bin."],
                    "step_instr": ["Go to the potato near the sink", "Pick up the potato", "Go to the microwave next to the fridge.", "Open the microwave", "Cook the potato in the microwave", "Take out the potato", "Go to the recycle bin", "Throw the potato in the recycle bin"],
                    "vis_objs": ["cup", "microwave", "fridge", "garbagecan"], 
                    "completed_plans": [("Navigation", "Countertop"),("PickupObject", "Potato"), ("Navigation", "Microwave")]
                }

    print("\n---------------Example Task----------------")
    for key, value in curr_task.items():
        print(f"{key}: {value}")


    hlp_generator = LLM_HLP_Generator(knn_data_path="knn_set.pkl", emb_model_name="paraphrase-MiniLM-L6-v2", debug=True)

    generated_plan = hlp_generator.generate_hlp(curr_task, k=9)

    print("\n---------GPT3 generated HLP-------------")
    print(generated_plan)

    # %%

    for plan_list in generated_plan:
        plan=get_plan(plan_list)
        plot_trajectories(plan,task=curr_task['task_instr'],start=plan[0][0],end=np.unique([plan[i][-1] for i in range(len(plan))]))

# %%



