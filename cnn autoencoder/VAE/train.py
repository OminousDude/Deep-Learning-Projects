import torch
import torchvision.datasets as datasets  # Standard datasets
from tqdm import tqdm
from torch import nn, optim
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from dataset import MNISTDataModule
import pytorch_lightning as pl
from model import VAEpl
from utils import inference
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DeepSpeedStrategy
torch.set_float32_matmul_precision("medium")
pl.seed_everything(42)

# things to add
lr = 1e-4
batch_size = 32
num_workers = 2
model = VAEpl(lr)
dm = MNISTDataModule(batch_size, num_workers)
logger = TensorBoardLogger("my_checkpoint", name="scheduler_autolr_vae_pl_model")

# add callback for learning rate monitor, model checkpoint, and scheduler on plateau
callbacks = [pl.callbacks.LearningRateMonitor(logging_interval="step"),
             pl.callbacks.ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min", save_last=True),
             ]

if __name__ == "__main__":
    trainer = pl.Trainer(
        max_epochs=25,
        accelerator="gpu",
        devices=1,
        logger=logger,
        #precision=16,
        # strategy=DeepSpeedStrategy(
        #     stage=0,
        # ),
    )

    # trainer.tune(model, dm)
    trainer.fit(model, dm)

    inference(model, dm.mnist_test, 5, 5)