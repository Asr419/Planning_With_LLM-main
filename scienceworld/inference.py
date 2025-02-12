import torch
from transformers import LlamaForSequenceClassification, LlamaTokenizer

def predict_score(task_description, thought, action, observation, model_path):
    # Load the model and tokenizer
    model = LlamaForSequenceClassification.from_pretrained(model_path)
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    

    model.eval()
    

    input_text = f"Task: {task_description} Thought: {thought} Action: {action} Observation: {observation}"
    

    inputs = tokenizer.encode_plus(
        input_text,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    
    device = next(model.parameters()).device
    input_ids = input_ids
    attention_mask = attention_mask
    

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predicted_score = outputs.logits.item()
    
    return predicted_score


if __name__ == "__main__":
    model_path = "models/fine_tuned_llama_score_predictor"
    
    task_description = "Summarize a news article"
    thought = "I should read the article carefully and identify the main points"
    action = "Read the article and take notes on key information"
    observation = "The article discusses recent developments in renewable energy"
    
    score = predict_score(task_description, thought, action, observation, model_path)
    print(f"Predicted score: {score:.2f}")