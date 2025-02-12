import requests
import os
import json
import sys
import warnings
from transformers import GPT2Tokenizer
from dsp import LM
from dotenv import load_dotenv

CALL_COUNT_FILE = "call_counts.txt"

completion_tokens = prompt_tokens = 0
MAX_TOKENS = 4000
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
load_dotenv()

def read_call_count():
    try:
        with open(CALL_COUNT_FILE, "r") as file:
            return int(file.read())
    except FileNotFoundError:
        return 0

def write_call_count(count):
    with open(CALL_COUNT_FILE, "w") as file:
        file.write(str(count))



def tokens_in_text(text):
    """
    Accurately count the number of tokens in a string using the GPT-2 tokenizer.
    
    :param text: The input text.
    :return: The exact number of tokens in the text.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokens = tokenizer.encode(text)
    return len(tokens)


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
        




    def basic_request(self, prompt,n=1,stop=None, **kwargs):
        global completion_tokens, prompt_tokens
        # Prompt = f"There is a 5x5 maze and you are at (0,0). You need to reach the goal at (4,4). "
        self.json_data =  {
                        "dialogue":{
                            "question": prompt
                        },
                        "llm_metadata": {
                            "model_name": "gpt-4",
                            "temperature": 1.0,
                            "max_tokens": 100,
                            "top_p": 1,
                            "n" : n,
                            "frequency_penalty": 0,
                            "presence_penalty": 0,
                            "llm_type": "azure_chat_openai",
                            "stop":stop
                        },
                        
                    }
        
        outputs = []
        while n > 0:
            cnt = min(n, 20)
            n -= cnt
            self.json_data['llm_metadata']['n']=cnt
            openai_response = get_openai_response(self.azure_url, self.ims_org_id, self.api_key, self.temp_auth_token, self.json_data)
            if 'generations' in openai_response:
                openai_response['choices']=openai_response['generations'][0]
                outputs.extend([choice["message"]["content"] for choice in openai_response["choices"]])
                # log completion tokens
                completion_tokens += openai_response['llm_output']["token_usage"]["completion_tokens"]
                prompt_tokens += openai_response['llm_output']["token_usage"]["prompt_tokens"]
            else:
                print(openai_response)
                outputs.extend(['Please modify the prompt flag raised due to content filter or content length'])
            
            
        return outputs

    def __call__(self, prompt,n=1,stop=None, only_completed=True, return_sorted=False, **kwargs):
        sys.stdout.flush()
        return self.basic_request(prompt,n=n,stop=stop,**kwargs)
    
def gpt_usage(backend="gpt-35-turbo"):
    global completion_tokens, prompt_tokens
    if backend == "gpt-4":
        cost = completion_tokens / 1000 * 0.06 + prompt_tokens / 1000 * 0.03
    elif backend == "gpt-35-turbo":
        cost = completion_tokens / 1000 * 0.002 + prompt_tokens / 1000 * 0.0015
    elif backend == "gpt-3.5-turbo-16k":
        cost = completion_tokens / 1000 * 0.004 + prompt_tokens / 1000 * 0.003
    return {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": cost}
    

def gpt(prompt,n=1, engine='text-davinci-003', max_tokens=200,stop=None,temperature=1.0):
    lm=FireLM()
    # call_count = read_call_count()
    # call_count += 1
    
    # write_call_count(call_count)
    if stop != None :
        stop=[stop]
    sys.stdout.flush()

    return lm(prompt,n=n, stop=stop)