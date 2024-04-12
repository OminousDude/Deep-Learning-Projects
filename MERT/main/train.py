import torch
import numpy as np
from tqdm import tqdm
import random
from matplotlib import pyplot as plt
import pandas as pd
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

class BertTrainer:
    def __init__(self, model, vocab, lr = 3e-5, weight_decay=0.01,
                betas=(0.9, 0.999), warmup_steps=10000, log_freq=1000, device = "cuda"):
        self.device = device
        self.model = torch.compile(model.to(device))

        self.vocab = vocab
        self.vocab_flip = dict((v,k) for k,v in vocab.items())
        
        self.pad_token = self.vocab[0]

        self.scaler = torch.cuda.amp.GradScaler()

        self.loss_graph = []

        self.optim = torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)

        # self.optim = torch.optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        # self.optim_schedule = ScheduledOptim(self.optim, EMB_DIM, n_warmup_steps=warmup_steps)

        self.criterion = torch.nn.NLLLoss(ignore_index=0)

        self.log_freq = log_freq
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
    
    def train(self, loader, epoch) -> None:
        self.iteration(epoch, loader)

    def test(self, loader, epoch, tokenizer) -> None:
        self.iteration(epoch, loader, tokenizer=tokenizer, train=False)

    def _plot(self, step_no_reset):
        temp = pd.DataFrame()
        temp['data'] = np.array(torch.as_tensor(self.loss_graph, device='cpu'))
        moving_average = temp['data'].rolling(window=5).mean()
        moving_average_large = temp['data'].rolling(window=20).mean()
        plt.plot(torch.as_tensor(self.loss_graph, device='cpu'), label="Original Data", color='black')
        plt.plot(torch.as_tensor(moving_average, device='cpu'), label="Scaled Data", color='pink')
        plt.plot(torch.as_tensor(moving_average_large, device='cpu'), label="Scaled Data_10", color='blue')
        plt.ylabel("loss")
        plt.savefig("saves/loss_" + str(step_no_reset) + ".png")

    def _step_check_large(self, step_no_reset, data_0, mask_lm_out_0):
        if step_no_reset % self.log_freq == 0:
            self._plot(step_no_reset)
            
            self.model.save(step_no_reset)

            word_count = 0
            data_str = ""
            for val in data_0:
                word = str(self.vocab[val.item()])
                if word == self.pad_token:
                    break

                data_str += word + " "
                word_count += 1
            print("Original Data: " + data_str)
            
            pred_str = ""
            for vals in mask_lm_out_0:
                if word_count == 0:
                    break
                pred_str += str(self.vocab[vals.argmax().item()]) + " "
                word_count -= 1
            print("Predicted Data: " + pred_str)

    def iteration(self, epoch, data_loader, tokenizer=None, train=True) -> None:
        loss_sum_epoch = 0
        loss_sum_steps = 0
        step = 0
        step_no_reset = 0
        loss_graph = []
        cola = 0

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

    def test(self, text: str, tokenizer: PreTrainedTokenizerFast, max_seq_len: int = 128) -> None:
        tokens = tokenizer.encode(text, return_tensors='pt')

        pad = torch.zeros(max_seq_len - tokens[0].shape[0])
        tokens = torch.cat((tokens[0], pad), 0).unsqueeze(0).int()

        preds = self.model.forward(tokens.to(self.device))

        word_count = 0
        data_str = ""
        for val in tokens[0]:
            word = str(self.vocab[val.item()])
            if word == self.pad_token:
                break

            data_str += word + " "
            word_count += 1
        print("Original Data: " + data_str)
        
        pred_str = ""
        for vals in preds[0]:
            if word_count == 0:
                break
            pred_str += str(self.vocab[vals.argmax().item()]) + " "
            word_count -= 1
        print("Predicted Data: " + pred_str)