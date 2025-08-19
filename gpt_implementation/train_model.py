import os
from torch import optim, nn, utils, Tensor
import torch as t
import lightning as L
from data import *
from model import *
from torch.utils.data import DataLoader

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

@dataclass
class TrainConfig:
    lr: float

class LightningTransformer(L.LightningModule):
    def __init__(self, model_config: DecoderConfig, train_config: TrainConfig):
        super().__init__()
        self.model_config = model_config
        self.train_config = train_config
        self.model = model_config.get_model()
        self.loss_fn = nn.CrossEntropyLoss()

    def training_step(self, batch):
        out = self.model(batch)
        reshaped_out = einops.rearrange(out, "b seq alph -> b alph seq")
        loss = self.loss_fn(reshaped_out[:, :, :-1], batch[:, 1:])
        self.log("train loss", loss)
        return loss

    def validation_step(self, batch):
        out = self.model(batch)
        reshaped_out = einops.rearrange(out, "b seq alph -> b alph seq")
        loss = self.loss_fn(reshaped_out[:, :, :-1], batch[:, 1:])
        self.log("validation loss", loss)
        return loss
 
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.train_config.lr)
        return optimizer

if __name__ == "__main__":

    seed_everything(42, workers=True)
    device = 'cuda' if t.cuda.is_available() else 'cpu'

    model_config = DecoderConfig(
        n_layers = 12,
        d_model = 768,
        n_heads = 8,
        d_head = 64,
        alphabet_size = 50257
    )

    train_config = TrainConfig(
        lr = 1e-2
    )

    version_num = 5
    
    logger = TensorBoardLogger(save_dir=os.getcwd(), version=version_num, name="lightning_logs")

    checkpoint_config = ModelCheckpoint(
        dirpath = f"./lightning_logs/version_{version_num}/checkpoints",
        filename = "my-model-{epoch:02d}",
        save_top_k = 1,
        save_on_train_epoch_end=True
    )


    transformer = LightningTransformer(model_config, train_config).to(device)
    train_dataset, val_dataset = get_datasets("./tiny_shakespeare.txt", 64)

    train_dataloader = DataLoader(train_dataset, batch_size = 64, shuffle = True, num_workers = 1)
    val_dataloader = DataLoader(val_dataset, batch_size = 64, shuffle = False, num_workers = 1)
    #print(t.cuda.is_available())
    #print(device)

    trainer = Trainer(
        accelerator = "gpu",
        devices = 1,
        log_every_n_steps = 10,
        logger = logger,
        max_epochs = 100,
        max_time = {"days": 1},
        val_check_interval = 50,
        callbacks = [checkpoint_config]
    )
    trainer.fit(transformer, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)