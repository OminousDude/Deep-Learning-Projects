import torch
import torch.nn as nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader, Subset

from torch.utils.tensorboard import SummaryWriter

class LightningModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.prepare_data()

        self._train_dataloader = self.train_dataloader()
        self._val_dataloader = self.val_dataloader()
        self._test_dataloader = self.test_dataloader()

        self.logger = SummaryWriter()

        self.current_epoch = 0
        self.current_step = 0
        self.max_step = 0

    def log(self, name: str, val: "int | float | torch.tensor") -> None:
        print(name + ": " + val.__str__())

    def forward(self, x):
        pass

    def prepare_data(self):
        pass

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass

    def configure_optimizers(self):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def on_validation_epoch_end(self):
        pass

    def on_fit_start(self):
        pass

    def on_fit_end(self):
        pass

    def on_save_checkpoint(self):
        pass

    def on_train_epoch_end(self):
        pass

    def on_train_batch_start(self):
        pass

    def load(self, check_loc = ""):
        if check_loc != "":
            torch.load(check_loc)
            self.eval()

        # state_dict = torch.load(self.check_loc)

        # model.load_state_dict(state_dict)


class Trainer:
    def __init__(self, max_epochs = 5, min_epochs = 0, accelerator = 'cpu', precision = '32', checkpoint_freq = 0, checkpoint_loc = ""):
        super().__init__()

        self.epochs = max_epochs
        self.accelerator = accelerator

        self.check_loc = checkpoint_loc
        self.check_freq = checkpoint_freq

        self.optim = None

        self.precision = None
        self.mixed = True
        if precision == '32':
            self.precision = torch.float32
        elif precision == '16':
            self.precision = torch.float16
        elif precision == 'bf-16-true':
            self.precision = torch.bfloat16
            self.mixed = False
        elif precision == 'bf-16-mixed':
            self.precision = torch.bfloat16


    def __train_loop(self, model: LightningModule):
        scaler = torch.cuda.amp.GradScaler()

        gpu = self.accelerator == 'gpu' or self.accelerator == 'cuda'
        if gpu:
            model.to('cuda')

        model.train()
        
        for i, (X, y) in enumerate(model._train_dataloader):
            self.optim.zero_grad()

            model.current_step = i
            model.on_train_batch_start()
            if gpu:
                X = X.to('cuda')
                y = y.to('cuda')

            with torch.cuda.amp.autocast(enabled=self.mixed, dtype=self.precision):
                loss = model.training_step((X, y), i)["loss"]

            scaler.scale(loss).backward()

            scaler.step(self.optim)

            scaler.update()

        model.on_train_epoch_end()

    def __val_loop(self, model: LightningModule):
        gpu = self.accelerator == 'gpu' or self.accelerator == 'cuda'
        if gpu:
            model.to('cuda')

        model.eval()
        with torch.no_grad():
            for i, (X, y) in enumerate(model._val_dataloader):
                if gpu:
                    X = X.to('cuda')
                    y = y.to('cuda')

                print("Loss " + i.__str__() + ": " + model.validation_step((X, y), i)["loss"].__str__())
        
        model.on_validation_epoch_end()

    def __test_loop(self, model: LightningModule):
        gpu = self.accelerator == 'gpu' or self.accelerator == 'cuda'
        if gpu:
            model.to('cuda')

        model.eval()
        test_loss = 0

        with torch.no_grad():
            for i, (X, y) in enumerate(model._test_dataloader):
                if gpu:
                    X = X.to('cuda')
                    y = y.to('cuda')
                print("Loss " + i.__str__() + ": " + model.test_step((X, y), i)["test_loss"].__str__())
        
    def fit(self, model):
        model.on_fit_start()

        model.max_step = len(model._train_dataloader)

        self.optim = model.configure_optimizers()

        model.current_epoch = 0

        for epoch in range(self.epochs):
            model.current_epoch = epoch
            if self.check_loc != "" and self.check_freq > 0 and epoch % self.check_freq == 0:
                model.on_save_checkpoint()
                torch.save(model, self.check_loc)
            self.__train_loop(model)
    
        model.on_fit_end()

    def validate(self, model):
        model.current_epoch = 0
        for epoch in range(self.epochs):
            model.current_epoch = epoch
            self.__val_loop(model)

    def test(self, model):
        self.__test_loop(model)

def seed_everything(seed):
    torch.manual_seed(seed)