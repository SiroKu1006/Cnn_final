import os
import argparse
import random
from typing import Optional, Union
from tqdm import tqdm

import torchaudio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold

from hmmlearn import hmm


class ASRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = nn.Conv2d(1, 32, kernel_size=(2, 2), padding="same")
        self.active_1 = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.drop_1 = nn.Dropout(0.35)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(32 * 20 * 50, 128)
        self.active_2 = nn.ReLU()
        self.drop_2 = nn.Dropout(0.25)
        self.classify = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.conv2d(x)
        x = self.active_1(x)
        x = self.max_pool(x)
        x = self.drop_1(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.active_2(x)
        x = self.drop_2(x)
        y = self.classify(x)
        
        return y


class ASRDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.target_num_frames = 100
        self.data = self.initialize_data()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        audio_file_path = self.data[index][0]
        waveform, sample_rate = torchaudio.load(audio_file_path)
        
        time_mask = torchaudio.transforms.TimeMasking(time_mask_param=5)
        freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=2)
        mfcc_transform = torchaudio.transforms.MFCC(
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 64}
        )
        
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)

        mfcc = mfcc_transform(waveform)
        mfcc = mfcc[0]
        
        if random.random() < 0.5:
            mfcc = time_mask(mfcc)
        if random.random() < 0.5:
            mfcc = freq_mask(mfcc)
        
        # Pad/truncate along the time dimension
        time_frames = mfcc.shape[1]

        if time_frames > self.target_num_frames:
            # Truncate
            mfcc = mfcc[:, :self.target_num_frames]
        else:
            # Pad
            pad_amount = self.target_num_frames - time_frames
            # F.pad expects (batch, channels, ...)
            mfcc = mfcc.unsqueeze(0)  # now shape: [1, n_mfcc, time_frames]
            mfcc = torch.nn.functional.pad(mfcc, (0, pad_amount))  # pad on the time dimension
            mfcc = mfcc.squeeze(0)  # back to [n_mfcc, target_num_frames]
        
        label = nn.functional.one_hot(torch.tensor(self.data[index][1]), num_classes=10).to(torch.float32)
        
        return mfcc, label

    def initialize_data(self):
        data = []
        for root, _, files in os.walk("./語音資料0_9"):
            for file in files:
                if file.endswith(".wav"):
                    label = int(file[0])
                    data.append([os.path.join(root, file), label])
        
        return data

class Trainer:
    def __init__(
        self,
        device: Union[str, torch.device],
        model: nn.Module,
        optimizer: optim.Optimizer,
        loss_fn: nn.Module,
        train_data: Optional[DataLoader] = None,
        valid_data: Optional[DataLoader] = None,
        test_data: Optional[DataLoader] = None,
    ):
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
    
    def set_train_data(self, train_data):
        self.train_data = train_data
    
    def set_valid_data(self, valid_data):
        self.valid_data = valid_data
    
    def set_test_data(self, test_data):
        self.test_data = test_data
    
    def train(self):
        self.model.train()
        
        total_train_loss = 0.0
        pbar = tqdm(desc="Training", total=len(self.train_data), dynamic_ncols=True)
        for batch in self.train_data:
            batch = self.batch_to_device(batch)
            input_ids = batch[0]
            input_ids = input_ids.unsqueeze(1)
            label = batch[1]
            
            hypo = self.model(input_ids)
            hypo = F.log_softmax(hypo, dim=1)
            label = label.argmax(dim=1)
            loss = self.loss_fn(hypo, label)
            
            loss.backward()
            total_train_loss += loss.item()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            pbar.update(1)
        pbar.close()
        
        valid_loss, valid_acc = self.valid()
        
        return total_train_loss / len(self.train_data), valid_loss, valid_acc

    def batch_to_device(self, batch):
        return [b.to(self.device) for b in batch]
    
    @torch.no_grad()
    def valid(self):
        self.model.eval()
        
        acc = 0.0
        loss = 0.0
        pbar = tqdm(desc="Validating", total=len(self.valid_data), dynamic_ncols=True)
        for batch in self.valid_data:
            batch = self.batch_to_device(batch)
            input_ids = batch[0]
            input_ids = input_ids.unsqueeze(1)
            label = batch[1]
            
            hypo = self.model(input_ids)
            hypo = F.log_softmax(hypo, dim=1)
            label = label.argmax(dim=1)
            loss += self.loss_fn(hypo, label)
            
            acc += (hypo.argmax(dim=1) == label).sum().item() / len(label)
            pbar.update(1)
        pbar.close()
        
        return loss / len(self.valid_data), acc / len(self.valid_data)

def train_nn():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = ASRNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    loss_fn = nn.NLLLoss()
    
    dataset = ASRDataset()
    kf = KFold(n_splits=5,shuffle=True)

    trainer = Trainer(device, model, optimizer, loss_fn)
    for epoch in range(100):
        for index, (train_index, valid_index) in enumerate(kf.split(dataset)):
            print(f"Fold: {index + 1}")
            train_data = DataLoader(dataset, batch_size=60, sampler=train_index, pin_memory=True)
            valid_data = DataLoader(dataset, batch_size=60, sampler=valid_index, pin_memory=True)
            
            trainer.set_train_data(train_data)
            trainer.set_valid_data(valid_data)
            train_loss, valid_loss, valid_acc = trainer.train()
            
            print(f"Epoch: {epoch}, Train Loss: {train_loss}, Valid Loss: {valid_loss}, Valid Acc: {valid_acc}")

def train_hmm():
    kmeans = KMeans(n_clusters=64, random_state=42)
    
    dataset = ASRDataset()
    dataloader = DataLoader(dataset)
    
    class_to_mfcc = {i: [] for i in range(10)}
    for batch in dataloader:
        fearture, label = batch
        scalar_label = label.squeeze(0).argmax().item()
        fearture = fearture.squeeze(0).numpy().T
        class_to_mfcc[scalar_label].append(fearture)
    
    all_mfcc = []
    for label, seq in class_to_mfcc.items():
        for arr in seq:
            all_mfcc.append(arr)
    
    flattened_mfcc = np.concatenate(all_mfcc, axis=0)
    kmeans = kmeans.fit(flattened_mfcc)
    
    class_to_discrete = {i: [] for i in range(10)}
    for label, seq in class_to_mfcc.items():
        for arr in seq:
            discrete = kmeans.predict(arr).reshape(-1, 1)
            class_to_discrete[label].append(discrete)
    
    label_to_hmm = {i: None for i in range(10)}
    for i in range(10):
        seq_list = class_to_discrete[i]
        x = np.concatenate(seq_list, axis=0)
        lengths = [len(s) for s in seq_list]
        hmm_model = hmm.CategoricalHMM(n_components=6, n_iter=100, random_state=42)
        hmm_model.fit(x, lengths)
        label_to_hmm[i] = hmm_model

def classify(
    kmeans,
    mfcc,
    hmm_models: dict[int, hmm.CategoricalHMM]
):
    discretes = kmeans.predict(mfcc).reshape(-1, 1)
    
    best_score = float("-inf")
    best_label = None
    for label, model in hmm_models.items():
        score = model.score(discretes)
        if score > best_score:
            best_score = score
            best_label = label
    
    return best_label

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--model", type=str, default="nn")
    args = arg_parser.parse_args()
    
    if args.model == "nn":
        train_nn()
    elif args.model == "hmm":
        train_hmm()
    else:
        raise ValueError("Invalid model type")