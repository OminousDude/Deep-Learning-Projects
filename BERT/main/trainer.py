import torch
import numpy as np
from tqdm import tqdm
import random
from matplotlib import pyplot as plt
import pandas as pd
from transformers import PreTrainedTokenizerBase
from torch.utils.data import DataLoader

from model import BertLM

class BertTrainer:
    def __init__(self, model: BertLM, vocab: dict, lr: float = 3e-5, weight_decay: float = 0.01,
                betas: tuple[float, float] = (0.9, 0.999), warmup_steps: int = 10000, log_freq: int = 1000, device: str = "cuda"):
        self.device = device
        # Compile model to speed up (around 50% boost in speed)
        self.model = torch.compile(model.to(device))

        self.vocab = vocab
        
        self.pad_token = self.vocab[0]

        self.scaler = torch.cuda.amp.GradScaler()

        self.loss_graph = []

        self.optim = torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)

        # self.optim = torch.optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        # self.optim_schedule = ScheduledOptim(self.optim, EMB_DIM, n_warmup_steps=warmup_steps)

        # Negative Log Likelihood (Likelihood a.k.a Cross Entropy)
        # "ignore_index" is what token index tro ignore because if padding is counted to total loss
        # it will throw off the actual loss and model will only predict padding
        self.criterion = torch.nn.NLLLoss(ignore_index=0)

        self.log_freq = log_freq
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
    
    def train(self, loader: DataLoader, epoch: int) -> None:
        self.iteration(epoch, loader)

    def test(self, loader: DataLoader, epoch: int, tokenizer: PreTrainedTokenizerBase) -> None:
        self.iteration(epoch, loader, tokenizer=tokenizer, train=False)

    def _plot(self, step_no_reset: int):
        # Plot the loss based on "loss_graph" list
        temp = pd.DataFrame()
        temp['data'] = np.array(torch.as_tensor(self.loss_graph, device='cpu'))
        moving_average_large = temp['data'].rolling(window=20).mean()
        plt.plot(torch.as_tensor(self.loss_graph, device='cpu'), label="Original Data", color='black')
        plt.plot(torch.as_tensor(moving_average_large, device='cpu'), label="Scaled Data_10", color='blue')
        plt.ylabel("loss")
        plt.savefig("saves/loss_" + str(step_no_reset) + ".png")

    def _step_check_large(self, step_no_reset: int, data: torch.Tensor, mask_lm_out: torch.Tensor):
        if step_no_reset % self.log_freq == 0:
            self._plot(step_no_reset)
            
            self.model.save(step_no_reset)

            self.print_no_padding(data, mask_lm_out)

    def iteration(self, epoch: int, data_loader: DataLoader, tokenizer: PreTrainedTokenizerBase = None, train: bool = True) -> None:
        loss_sum_epoch = 0
        loss_sum_steps = 0
        step = 0
        step_no_reset = 0
        loss_graph = []

        if train:
            self.model.train()
        else:
            self.model.eval()

        # Iterate through dataloader
        for data, labels in tqdm(data_loader):
            data = data.to(self.device)
            labels = labels.to(self.device)

            # Convert data to 16 bit
            with torch.cuda.amp.autocast():
                mask_lm_output = self.model.forward(data)
                
            # Perform cross entropy loss
            mask_loss = self.criterion(mask_lm_output.transpose(1, 2), labels)
            loss = mask_loss
            loss_sum_steps += loss
            loss_sum_epoch += loss

            # Train the model and normalize (scale) the data scaler
            if train:
                self.optim.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()

            # Create loss img and save the model
            self._step_check_large(step_no_reset, data[0], mask_lm_output[0])
            
            # Save a loss checkpoint
            if step % 100 == 0:
                loss_sum_steps = 0
                step = 0
                loss_graph.append(loss)
                print(loss)
            step += 1
            step_no_reset += 1

        print(
            f"EP{epoch}, AVG LOSS{loss_sum_epoch / step_no_reset}"
        )
        
        self.model.save(0, epoch)

        self._plot(step_no_reset)

    def print_no_padding(self, original, predictions):
        # Add the tokens to string until the first "[PAD]" is encountered in the original data then print my usefull (non-padding) tokens
        data_str = ""
        pred_str = ""
        for i in range(original.shape[0]):
            data_val = original[i]
            out_vals = predictions[i]
            word = str(self.vocab[data_val.item()])
            if word == self.pad_token:
                break

            data_str += word + " "

            pred_str += str(self.vocab[out_vals.argmax().item()]) + " "
        print("Original Data: " + data_str)
        print("Predicted Data: " + pred_str)

    def test(self, text: str, tokenizer: PreTrainedTokenizerBase, max_seq_len: int = 128) -> None:
        # Tokenize input sentence
        tokens = tokenizer.encode(text, return_tensors='pt')

        # Pad the sentence
        pad = torch.zeros(max_seq_len - tokens[0].shape[0])
        tokens = torch.cat((tokens[0], pad), 0).unsqueeze(0).int()

        # Predict the outputs
        preds = self.model.forward(tokens.to(self.device))

        self.print_no_padding(tokens, preds)