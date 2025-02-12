import os
import re
from base import Task
from transformers import GPT2Tokenizer
import random
from models_custom import gpt
from scworld import *

import torch
from transformers import LlamaForSequenceClassification, LlamaTokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
#OLLAMA-3b
model_path = "hotpot_qa/trained_model"
model = LlamaForSequenceClassification.from_pretrained(model_path)
tokenizer = LlamaTokenizer.from_pretrained(model_path)
#LLAMA2-7b
model_path_llama = "hotpot_qa/llama_trained_model"
model_llama = LlamaForSequenceClassification.from_pretrained(model_path_llama)
tokenizer_llama = LlamaTokenizer.from_pretrained(model_path_llama)

def get_token_length(text):
    return len(tokenizer.encode(text))

max_token_length = 4000

class ScWorldTask(Task):
    def __init__(self):
        """
        file: a text file, each line is some sentences
        """
        super().__init__()
        self.steps = 7
        self.stops = ['\nObservation:\n', None]
        self.value_cache = {}

    def __len__(self) -> int:
        return len(self.data)
    
    @staticmethod
    def standard_prompt_wrap(x: str, y:str='') -> str:
        return standard_prompt.format(input=x) + y
    

    @staticmethod
    def compare_output_unwrap(compare_output: str):
        if 'more correct trajectory is 1' in compare_output:
            return 0
        elif 'more correct trajectory is 2' in compare_output:
            return 1
        elif "two trajectories are similarly correct" in compare_output:
            return 0.5
        else:
            print(f'-----------------compare no match: {[compare_output]}')
            return -1

    @staticmethod
    def compare_prompt_wrap(x: str, ys: list) -> str:
        assert len(ys) == 2, 'compare prompt only supports 2 candidates'
        
        # Extract the last Action for each trajectory
        last_actions = []
        for y in ys:
            # Split by line and reverse to start from the end
            lines = y.split('\n')[::-1]
            for line in lines:
                # Check for an Action line and get its content
                if "Action" in line:
                    last_actions.append(line.split('Action')[-1].strip(': '))
                    break

        assert len(last_actions) == 2, 'Expected to find 2 Actions'

        # Construct the prompt with the extracted Actions
        prompt = compare_prompt + f'Action 1:{last_actions[0]}\n\nAction 2:{last_actions[1]}\n'
        return prompt


    @staticmethod
    def vote_prompt_wrap(x: str, ys: list) -> str:
        prompt = vote_prompt + "\n" + x + "\n\n"
        for i, y in enumerate(ys, 1):
            # y = y.replace('Plan:\n', '')
            # TODO: truncate the plan part?
            prompt += f'Choice {i}:\n{y}\n'
        return prompt
    
    @staticmethod
    def value_prompt_wrap(x: str, y: str) -> str:
        inp = y + "\nThis trajectory is "
        #inp = y + "\nThus the correctess score is "
        #prompt = value_prompt.format(s="", input=inp)
        prompt = value_prompt_reasoning.format(s="", input=inp)
            
        return prompt
    
    @staticmethod
    def value_outputs_unwrap(evaluate_prompt: str):
        evaluate_prompt = evaluate_prompt[0]
        if '10' in evaluate_prompt:
            return 1.0
        elif '9' in evaluate_prompt:
            return 0.9
        elif '8' in evaluate_prompt:
            return 0.8
        elif '7' in evaluate_prompt:
            return 0.7
        elif '6' in evaluate_prompt:
            return 0.6
        elif '5' in evaluate_prompt:
            return 0.5
        elif '4' in evaluate_prompt:
            return 0.4
        elif '3' in evaluate_prompt:
            return 0.3
        elif '2' in evaluate_prompt:
            return 0.2
        elif '1' in evaluate_prompt:
            return 0.1
        else:
            return -1

    def predict_score(task_description, thought, action, observation):
    # Load the model and tokenizer
        
        
        # Set the model to evaluation mode
        model_llama.eval()
        
        # Prepare the input text
        input_text = f"Task: {task_description} Thought: {thought} Action: {action} Observation: {observation}"
        
        # Tokenize the input
        inputs = tokenizer_llama.encode_plus(
            input_text,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        # Make sure we're using the same device as the model
        device = next(model_llama.parameters()).device
        input_ids = input_ids
        attention_mask = attention_mask
        
        # Predict the score
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predicted_score = outputs.logits.item()
    
        return predicted_score
    
    @staticmethod
    def trained_llm_scoring(text:str):
        
        task_description = re.search(r'(.*?)(?=\nThought 1:)', text, re.DOTALL).group(1).strip()

        thought = re.findall(r'Thought \d+: (.*)', text)[-1].strip()
        action = re.findall(r'Action \d+: (.*)', text)[-1].strip()
        observation = re.findall(r'Observation \d+: (.*)', text, re.DOTALL)[-1].strip()

        score = ScWorldTask.predict_score(task_description, thought, action, observation)

        return score