import os
import math
import random
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import Transformer
from data import build_vocab,process_sentence,TranslationDataset,collate_fn

import yaml
import torch

    def load_config(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    # 加载配置
    config = load_config('/config/base.yaml')

    # 使用配置中的超参数
    DATA_PATH = config['DATA_PATH']
    DEVICE = torch.device("cuda" if torch.cuda.is_available() and config['DEVICE'] == 'auto' else "cpu")
    BATCH_SIZE = config['BATCH_SIZE']
    D_MODEL = config['D_MODEL']
    NUM_HEADS = config['NUM_HEADS']
    D_FF = config['D_FF']
    NUM_LAYERS = config['NUM_LAYERS']
    DROPOUT = config['DROPOUT']
    LR = config['LR']
    N_EPOCHS = config['N_EPOCHS']
    MAX_POSITION = config['MAX_POSITION']
    SEED = config['SEED']




    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"请把数据集放到 {DATA_PATH}，文件每行格式应为: english \\t chinese")

    pairs = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                en = parts[0].strip()
                zh = parts[1].strip()
                if len(en) > 0 and len(zh) > 0:
                    pairs.append((en, zh))

    df = pd.DataFrame(pairs, columns=["English", "Chinese"])
    print("样例数据：")
    print(df.head())



    # tokenizers
    tokenizer_en = get_tokenizer("basic_english")
    tokenizer_zh = lambda x: list(x)

    en_sentences = df["English"].tolist()
    zh_sentences = df["Chinese"].tolist()



    en_vocab = build_vocab(en_sentences, tokenizer_en)
    zh_vocab = build_vocab(zh_sentences, tokenizer_zh)
    print(f"vocab sizes -> en: {len(en_vocab)}, zh: {len(zh_vocab)}")



    en_sequences = [process_sentence(s, tokenizer_en, en_vocab) for s in en_sentences]
    zh_sequences = [process_sentence(s, tokenizer_zh, zh_vocab) for s in zh_sentences]


    dataset = TranslationDataset(en_sequences, zh_sequences)
    train_set, val_set = train_test_split(list(range(len(dataset))), test_size=0.1, random_state=SEED)
    train_dataset = torch.utils.data.Subset(dataset, train_set)
    val_dataset = torch.utils.data.Subset(dataset, val_set)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)


    model = Transformer(
        src_vocab_size=len(en_vocab),
        trg_vocab_size=len(zh_vocab),
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=zh_vocab["<pad>"])
    optimizer = optim.Adam(model.parameters(), lr=LR)

    def train_one_epoch(model, dataloader, optimizer, criterion, device):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc="Train batches", leave=False)
        for src, trg in pbar:
            src = src.to(device)
            trg = trg.to(device)
            optimizer.zero_grad()
            # input to decoder excludes last token
            output = model(src, trg[:, :-1])  # [batch, trg_len-1, vocab]
            out_dim = output.size(-1)
            output_flat = output.contiguous().view(-1, out_dim)
            trg_flat = trg[:, 1:].contiguous().view(-1)
            loss = criterion(output_flat, trg_flat)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        return epoch_loss / len(dataloader)

    def evaluate(model, dataloader, criterion, device):
        model.eval()
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc="Val batches", leave=False)
        with torch.no_grad():
            for src, trg in pbar:
                src = src.to(device)
                trg = trg.to(device)
                output = model(src, trg[:, :-1])
                out_dim = output.size(-1)
                output_flat = output.contiguous().view(-1, out_dim)
                trg_flat = trg[:, 1:].contiguous().view(-1)
                loss = criterion(output_flat, trg_flat)
                epoch_loss += loss.item()
                pbar.set_postfix({"val_loss": f"{loss.item():.4f}"})
        return epoch_loss / len(dataloader)


    train_losses = []
    val_losses = []

    print("Start training...")
    for epoch in range(1, N_EPOCHS + 1):
        start_time = time.time()
        # Training
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        # Validation
        val_loss = evaluate(model, val_loader, criterion, DEVICE)
        elapsed = time.time() - start_time

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch}/{N_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {elapsed:.1f}s")

    plt.figure(figsize=(8,5))
    plt.plot(range(1, N_EPOCHS+1), train_losses, label="Train Loss")
    plt.plot(range(1, N_EPOCHS+1), val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()