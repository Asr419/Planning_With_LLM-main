import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import LlamaForSequenceClassification, LlamaTokenizer, get_linear_schedule_with_warmup, AutoTokenizer
from torch.optim import AdamW
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
import torch.nn as nn
import wandb
from torch.cuda.amp import GradScaler

from accelerate import Accelerator






class ScoreDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        input_text = f"Task: {item['task_description']} Thought: {item['thought']} Action: {item['action']} Observation: {item['observation']}"
        inputs = self.tokenizer.encode_plus(
            input_text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'score': torch.tensor(item['score'], dtype=torch.bfloat16)
        }
    

class Inference():
    def predict_score(task_description, thought, action, observation):
        model = LlamaForSequenceClassification.from_pretrained(model_save_path)
        tokenizer = LlamaTokenizer.from_pretrained(model_save_path)
        model.eval()

        input_text = f"Task: {task_description} Thought: {thought} Action: {action} Observation: {observation}"
        inputs = tokenizer.encode_plus(
            input_text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predicted_score = outputs.logits.item()

        return predicted_score
    
if __name__=="__main__":
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"
   
    data = pd.read_csv('pretraining_data/data.csv')
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

    # model_name="openlm-research/open_llama_3b_v2"
    model_name = "meta-llama/Llama-2-7b-hf"
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    model = LlamaForSequenceClassification.from_pretrained(model_name, num_labels=1, torch_dtype=torch.bfloat16)
   

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    tokenizer.padding_side = "right"

    # Prepare datasets and dataloaders
    max_length = 512
    train_dataset = ScoreDataset(train_data, tokenizer, max_length)
    val_dataset = ScoreDataset(val_data, tokenizer, max_length)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1)

   
    # device='cpu'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    optimizer = AdamW(model.parameters(), lr=2e-5)
    num_epochs = 30
    num_training_steps = num_epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    mse_loss = nn.MSELoss().to(torch.bfloat16)
    accelerator=Accelerator(mixed_precision='bf16')

    model,optimizer,train_loader,val_loader=accelerator.prepare(model,optimizer,train_loader,val_loader)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            scores = batch['score']

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # print(f"outputs : {outputs}")
            logits = outputs.logits.view(-1).to(torch.bfloat16)
            scores=scores.view(-1).to(torch.bfloat16)

            # print(f"Scores: {scores}")
           
            loss = mse_loss(logits, scores)

            total_loss += loss.item()
            # loss.backward()
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Average train loss: {avg_train_loss:.4f}")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                scores = batch['score']

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits.view(-1) 
                scores=scores.view(-1)
                loss = mse_loss(logits, scores)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
   
        print(f"Validation loss: {avg_val_loss:.4f}")

    # Save the fine-tuned model
    if accelerator.is_main_process:
        # model.save_pretrained("fine_tuned_llama_score_predictor")
        # tokenizer.save_pretrained("fine_tuned_llama_score_predictor")
        os.makedirs('models', exist_ok=True)
        model_save_path = os.path.join('models', 'fine_tuned_llama_2_7b_score_predictor_2')
        model.module.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)


    # # Save the fine-tuned model
    

    
