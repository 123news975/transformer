
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

## 分词相关操作和数据集定义

def build_vocab(sentences, tokenizer):
    def yield_tokens():
        for s in sentences:
            yield tokenizer(s)
    vocab = build_vocab_from_iterator(yield_tokens(), specials=["<unk>", "<pad>", "<bos>", "<eos>"])
    vocab.set_default_index(vocab["<unk>"])
    return vocab


def process_sentence(sent, tokenizer, vocab):
    tokens = ["<bos>"] + tokenizer(sent) + ["<eos>"]
    return [vocab[t] for t in tokens]

# dataset
class TranslationDataset(Dataset):
    def __init__(self, src_list, trg_list):
        assert len(src_list) == len(trg_list)
        self.src_list = src_list
        self.trg_list = trg_list
    def __len__(self):
        return len(self.src_list)
    def __getitem__(self, idx):
        return torch.tensor(self.src_list[idx], dtype=torch.long), torch.tensor(self.trg_list[idx], dtype=torch.long)

def collate_fn(batch):
    srcs, trgs = zip(*batch)
    srcs = pad_sequence(srcs, batch_first=True, padding_value=en_vocab["<pad>"])
    trgs = pad_sequence(trgs, batch_first=True, padding_value=zh_vocab["<pad>"])
    return srcs, trgs

