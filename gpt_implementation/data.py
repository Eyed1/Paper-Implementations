import torch as t
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoTokenizer

class CustomTextDataset(Dataset):
    def __init__(self, text_file, seq_len):
        super().__init__()

        self.seq_len = seq_len
        self.text_file = text_file
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        self.tokens = []
        with open(text_file) as f:
            for i, line in enumerate(f.readlines()):
                self.tokens += tokenizer(line)["input_ids"]
        self.tokens = t.Tensor(self.tokens).long()
        
    def __len__(self):
        return len(self.tokens)//self.seq_len 
        
    def __getitem__(self, idx):
        return self.tokens[idx*self.seq_len:(idx+1)*self.seq_len]

def get_datasets(text_file, seq_len, train_pct = 0.7):
    dataset = CustomTextDataset(text_file, seq_len)

    train_size = int(len(dataset)*train_pct)
    train_dataset = Subset(dataset, range(train_size))
    val_dataset = Subset(dataset, range(train_size, len(dataset)))

    return train_dataset, val_dataset


if __name__ == "__main__":
    #tmp_dataset = CustomTextDataset("./tiny_shakespeare.txt", 64)

    train_dataset, val_dataset = get_datasets("./tiny_shakespeare.txt", 256, 0.5)
    print(len(train_dataset))
    print(len(val_dataset))
    print(train_dataset[0])
