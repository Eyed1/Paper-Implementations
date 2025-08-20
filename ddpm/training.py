import os
import numpy as np
from dataclasses import dataclass

from torch import optim, nn, utils, Tensor
from torch.utils.data import DataLoader, Dataset, Subset

import torch as t
import lightning as L
from load_data import *
from model import *
from loss import * 

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

@dataclass
class TrainConfig:
    lr: float
    beta_vals: list[float]
    T: int

class Lightning_ddpm(L.LightningModule):
    def __init__(self, train_config: TrainConfig, model: nn.Module, num_samples = 1):
        super().__init__()
        self.train_config = train_config

        alpha_vals = [1]
        alpha_prod_vals = [1]
        sigma_vals = [0]

        for i in range(self.train_config.T):
            alpha_vals.append(1 - self.train_config.beta_vals[i+1])
            alpha_prod_vals.append(alpha_prod_vals[i]*alpha_vals[i+1])
            sigma_vals.append(((1 - alpha_prod_vals[i])/(1-alpha_prod_vals[i+1]) \
                                    * train_config.beta_vals[i+1])**0.5)
        
        alpha_vals = t.Tensor(alpha_vals)
        alpha_prod_vals = t.Tensor(alpha_prod_vals)
        sigma_vals = t.Tensor(sigma_vals)

        #alpha_vals = t.cat(alpha_vals, 0)
        #alpha_prod_vals = t.cat(alpha_prod_vals,0)
        #sigma_vals = t.cat(sigma_vals,0)

        self.register_buffer("alpha_vals",      t.tensor(alpha_vals, dtype=t.float32), persistent=True)
        self.register_buffer("alpha_prod_vals", t.tensor(alpha_prod_vals, dtype=t.float32), persistent=True)
        self.register_buffer("sigma_vals",      t.tensor(sigma_vals, dtype=t.float32), persistent=True)

        device = 'cuda' if t.cuda.is_available() else 'cpu'
        self.num_samples = num_samples
        self.sample_batches = t.randn(num_samples, 3, 32, 32).to(device)

        self.model = model

    def sample(self):
        #device = 'cuda' if t.cuda.is_available() else 'cpu'
        xT = self.sample_batches 
        device = xT.device
        for t_val in range(self.train_config.T, 0, -1):
            z = 0
            if t_val > 1:
                z = t.randn(self.num_samples, 3,32,32).to(device)
            eps_pred = self.model(xT, t.Tensor([t_val]*self.num_samples).to(device))
            xT = 1/(self.alpha_vals[t_val]**0.5) * (xT - \
                (1 - self.alpha_vals[t_val])/((1-self.alpha_prod_vals[t_val])**0.5) * eps_pred) + \
                self.sigma_vals[t_val]*z
        return xT.cpu().detach().numpy()


    def training_step(self, batch):
        data = batch[0]
        t_sample = t.randint(low = 1, high = self.train_config.T+1, size = (data.shape[0],)).to(data.device)
        eps = t.randn(data.shape).to(data.device)
        alpha_prod = t.gather(self.alpha_prod_vals, dim = 0, index = t_sample)

        #print(alpha_prod.device, self.alpha_prod_vals.device, t_sample.device)

        pred_eps = self.model((alpha_prod**0.5).view(alpha_prod.shape[0], 1, 1, 1) * data \
                              + ((1 - alpha_prod)**0.5).view(alpha_prod.shape[0], 1, 1, 1) *eps, t_sample)
        loss = lower_t_l2_loss(eps, pred_eps)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        data = batch[0]
        t_sample = t.randint(low = 1, high = self.train_config.T+1, size = (data.shape[0],)).to(data.device)
        eps = t.randn(data.shape).to(data.device)
        alpha_prod = t.gather(self.alpha_prod_vals, dim = 0, index = t_sample)

        #print(alpha_prod.device, self.alpha_prod_vals.device, t_sample.device)

        pred_eps = self.model((alpha_prod**0.5).view(alpha_prod.shape[0], 1, 1, 1) * data \
                              + ((1 - alpha_prod)**0.5).view(alpha_prod.shape[0], 1, 1, 1) *eps, t_sample)
        loss = lower_t_l2_loss(eps, pred_eps)

        if batch_idx == 0:
            img: np.ndarray = self.sample()  # CHW

            # TensorBoard expects float in [0,1] or uint8 in [0,255].
            # Adjust this to your pipeline (examples shown):
            if img.dtype != np.uint8:
                # If in [-1, 1], map to [0,1]:
                # img = (img + 1.0) / 2.0
                # If in [0,255], map to [0,1]:
                img = (img+1)/2
                img = np.clip(img, 0.0, 1.0).astype(np.float32)

            for i in range(self.num_samples):

                # Avoid duplicate logs on multi-GPU
                if self.trainer.is_global_zero:
                    self.logger.experiment.add_image(
                        tag=f"val/sample{i}",
                        img_tensor=img[i],                 # can be numpy or torch.Tensor
                        global_step=self.global_step,   # or self.trainer.global_step
                        dataformats="CHW",              # important for (3, H, W)
                    )

        self.log("val/loss", loss)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.train_config.lr)
        return optimizer

if __name__ == "__main__":
    seed_everything(42, workers=True)
    device = 'cuda' if t.cuda.is_available() else 'cpu'

    ### Configs
    train_config = TrainConfig(
        lr = 2*1e-4,
        beta_vals = [0] + list(np.linspace(1e-4, 0.02, num = 10000)),
        T = 1000
    )

    version_num = 2
    logger = TensorBoardLogger(save_dir=os.getcwd(), version=version_num, name="lightning_logs")
    checkpoint_config = ModelCheckpoint(
        dirpath = f"./lightning_logs/version_{version_num}/checkpoints",
        filename = "my-model-{epoch:02d}",
        save_top_k = 1,
        monitor = "val/loss",
        save_last=True
    )


    ### Dataloading
    filedir = "/workspace/Paper-Implementations/ddpm/data/cifar-10-batches-py/"
    train_filepaths = [f"{filedir}/data_batch_{i}" for i in range(1, 6)]
    val_filepaths = [f"{filedir}/test_batch"]

    train_dataset= cifar_dataset(train_filepaths)
    val_dataset = cifar_dataset(val_filepaths)
    train_dataloader = DataLoader(train_dataset, batch_size = 128, shuffle = True, num_workers = 1)
    val_dataloader = DataLoader(val_dataset, batch_size = 128, shuffle = False, num_workers = 1)


    base_model = diffusion_model()
    lightning_model = Lightning_ddpm(train_config, base_model, 10)
    lightning_model.to(device)

    trainer = Trainer(
        accelerator = "gpu",
        devices = 1,
        log_every_n_steps = 10,
        logger = logger,
        limit_val_batches = 0.2,
        max_epochs = 500,
        max_time = {"days": 1},
        val_check_interval = 100,
        callbacks = [checkpoint_config]
    )
    trainer.fit(lightning_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    """
    print(train_dataset[0])
    print(lightning_model.sigma_vals)

    sample_res = lightning_model.sample()
    plt.imshow(sample_res[0].cpu().detach().numpy().transpose((1,2,0)))
    plt.show()
    """

            

