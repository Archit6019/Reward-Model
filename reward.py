import os 
import pandas as pd 
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from transformers import GPT2Model, GPT2Tokenizer

#--dataset formatting 
def preprocess(row):
    if row['choose-best']['value'][0]==2:
        row['response-1'], row['response-2'] = row['response-2'], row['response-1']
    return row 

#--loss function 
def loss(preferred_reward, alternate_reward):
    return -torch.mean(torch.log(torch.sigmoid(alternate_reward - preferred_reward)))

#--Reward Model Class 

class Reward_Model:
    def __init__(self):
        # super().__init__()
        dataset = load_dataset("argilla/reward-model-data-falcon")
        self.dataset = dataset.map(lambda x : preprocess(x))

    class GPT2(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = GPT2Model.from_pretrained("gpt2")
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.linear_head = torch.nn.Linear(768, 1)

        def forward(self, context, response):
            inp = context + response 
            inp_dict = self.tokenizer(
                '<|startoftext|>' + inp + '<|endoftext|>',
                return_tensors='pt'
            )
            input_ids = inp_dict.input_ids
            attention_mask = inp_dict.attention_mask

            model_outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            all_outputs = model_outputs.last_hidden_state
            last_output = all_outputs[:, -1, :]
            reward = self.linear_head(last_output)

            return reward
        
    def save_model(self, model, model_name : str):
            if not os.path.exists(model_name):
                os.makedirs(model_name)
            torch.save(model.state_dict(), f"{model_name}/model_state.pt")
            print(f"Model saved to {model_name}")

    def train(self, model : GPT2Model, epochs : int):
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
            batch_idx = 0
            for epoch in range(epochs):
                print(f"Epoch : {epoch + 1}")
                for batch in self.dataset:
                    prompt, preferred_reponse, alternate_response,choose,external_id = batch
                    preferred_reward = model(prompt, preferred_reponse)
                    alternate_reward = model(prompt, alternate_response)
                    loss_value = loss(preferred_reward, alternate_reward)
                    loss_value.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    print(f"Loss: {loss_value.item()}", batch_idx)
                    batch_idx += 1

            self.save_model(model, 'reward_model')

    @classmethod
    def load_model(cls, model_name : str):
            model = cls.GPT2()
            model.load_state_dict(torch.load(f"{model_name}/model_state.pt"))
            print(f"Model loaded from {model_name}")
            return model
        
    def inference(self, prompt : str, response_a : str, response_b : str, model : GPT2Model):
            model.eval()
            with torch.no_grad():
                reward_a = model(prompt, response_a)
                reward_b = model(prompt, response_b)
                difference = reward_a - reward_b
                preference_score = torch.sigmoid(difference)
            print(f"Reward for response A: {reward_a.item()}")
            print(f"Reward for response B: {reward_b.item()}")

            if preference_score > 0.5:
                print(f"The model prefers response A with {preference_score.item():.2%} confidence")
            elif preference_score < 0.5:
                print(f"The model prefers response B with {(1 - preference_score.item()):.2%} confidence")
            else:
                print("The model is neutral between the two responses")

