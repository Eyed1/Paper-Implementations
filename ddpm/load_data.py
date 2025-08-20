# %%
import torch as t
import pickle
from torch.utils.data import DataLoader, Dataset, Subset
import matplotlib.pyplot as plt
import einops

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

class cifar_dataset(Dataset):
    def __init__(self, filepaths = []):
        self.imgs = []
        self.labels = []
        #self.filenames = []
        for file in filepaths:
            dict = unpickle(file)
            self.imgs.append((t.Tensor(dict[b"data"]) - 127.5)/(127.5))
            self.labels.append(t.Tensor(dict[b"labels"]))
            #self.filenames.append(t.Tensor(dict[b"filenames"]))

        self.imgs = t.cat(self.imgs, 0)
        self.labels = t.cat(self.labels, 0)
        #self.filenames = t.cat(self.filenames, 0)

        self.imgs = einops.rearrange(self.imgs, "b (c h w) -> b c h w", h = 32, w = 32)

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        return self.imgs[idx], self.labels[idx]


# %%
filepath = "/workspace/Paper-Implementations/ddpm/data/cifar-10-batches-py/data_batch_1"
dict = unpickle(filepath)
#print(dict.keys())
#print(dict[b"filenames"])

test_arr = dict[b"data"][2]
reshaped_arr = einops.rearrange(test_arr, "(c h w) -> h w c", h = 32, w = 32)

plt.imshow(reshaped_arr)
plt.show()
# %%
filedir = "/workspace/Paper-Implementations/ddpm/data/cifar-10-batches-py/"
train_filepaths = [f"{filedir}/data_batch_{i}" for i in range(1, 6)]
val_filepaths = [f"{filedir}/test_batch"]

train_dataloader = cifar_dataset(train_filepaths)
val_dataloader = cifar_dataset(val_filepaths)
# %%
