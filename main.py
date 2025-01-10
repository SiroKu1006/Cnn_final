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

import matplotlib.pyplot as plt

import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold

from hmmlearn import hmm


class ASRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(2, 2), padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(0.2),
            nn.Flatten(),
            nn.Linear(32 * 20 * 50, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        y = self.net(x)
        
        return y


class ASRDataset(Dataset):
    def __init__(self, apply_augmentation: bool = False):
        super().__init__()
        self.apply_augmentation = apply_augmentation
        self.target_num_frames = 100
        self.data = self.initialize_data()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        audio_file_path = self.data[index][0]
        waveform, sample_rate = torchaudio.load(audio_file_path)
        
        if self.apply_augmentation:
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
        mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-6)
        
        if self.apply_augmentation:
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
            self.optimizer.step()
            self.optimizer.zero_grad()
            total_train_loss += loss.item()
            
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

def main(args):
    kf = KFold(n_splits=args.kfold, shuffle=True)
    dataset = ASRDataset(apply_augmentation=False)
    
    if args.model == "nn":
        train_nn(kf, dataset)
    elif args.model == "hmm":
        train_hmm(kf, dataset)
    else:
        raise ValueError("Invalid model type")

def train_nn(kf: KFold, dataset: ASRDataset):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    loss_fn = nn.NLLLoss()
    for index, (train_idx, valid_idx) in enumerate(kf.split(dataset)):
        acc_per_fold = []
        acc_per_fold_total = []
        valid_loss_per_fold = []
        valid_loss_per_fold_total = []
        print(f"Fold {index + 1}")
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        valid_subset = torch.utils.data.Subset(dataset, valid_idx)
        
        train_data = DataLoader(train_subset, batch_size=60, shuffle=True)
        valid_data = DataLoader(valid_subset, batch_size=60, shuffle=False)
        
        model = ASRNet().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        trainer = Trainer(device, model, optimizer, loss_fn, train_data=train_data, valid_data=valid_data)
        
        for epoch in range(15):
            train_loss, valid_loss, valid_acc = trainer.train()
            print(f"Epoch {epoch + 1}: Train loss: {train_loss}, Valid loss: {valid_loss}, Valid acc: {valid_acc}")
            valid_loss_per_fold.append(valid_loss)
            valid_loss_per_fold_total.append(valid_loss)
            acc_per_fold.append(valid_acc)
            acc_per_fold_total.append(valid_acc)
            if valid_acc >= max(acc_per_fold):
                torch.save(model.state_dict(), "best_model_10.pth")
            

    
        plt.title("Loss and acc")
        plt.plot(valid_loss_per_fold)
        plt.plot(acc_per_fold)
        plt.legend(["Loss", "Acc"])
        plt.xlabel("Epoch")
        # plt.savefig(f"nn_loss_acc{index+1}.png")
        # plt.show()
        plt.close()
        print(f"Average acc: {np.mean(acc_per_fold)}")
        print(f"Average valid loss: {np.mean(valid_loss_per_fold)}")
    print(f"Total Average acc: {np.mean(acc_per_fold_total)}")
    print(f"Total Average valid loss: {np.mean(valid_loss_per_fold_total)}")

def fit_kmeans(class_to_mfcc: dict[int, list[np.ndarray]]):
    print("Fitting kmeans")
    kmeans = KMeans(n_clusters=64, random_state=42)
    
    # Process data for kmeans
    all_mfcc = []
    for label, seq in class_to_mfcc.items():
        for arr in seq:
            all_mfcc.append(arr)
    
    flattened_mfcc = np.concatenate(all_mfcc, axis=0)
    kmeans = kmeans.fit(flattened_mfcc)
    
    return kmeans

def fit_dhmm(class_to_discrete: dict[int, list[np.ndarray]]):
    # Train DHMM
    label_to_hmm = {i: None for i in range(10)}
    for i in range(10):
        seq_list = class_to_discrete[i]
        x = np.concatenate(seq_list, axis=0).reshape(-1, 1)
        lengths = [len(s) for s in seq_list]
        hmm_model = hmm.CategoricalHMM(n_components=6, n_iter=100, random_state=42, n_features=64)
        hmm_model.fit(x, lengths)
        label_to_hmm[i] = hmm_model
    
    return label_to_hmm

def train_hmm(kf: KFold, dataset: ASRDataset):
    acc_per_fold = []
    for index, (train_idx, valid_idx) in enumerate(kf.split(dataset)):
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        valid_subset = torch.utils.data.Subset(dataset, valid_idx)
        
        train_data = DataLoader(train_subset, batch_size=1)
        valid_data = DataLoader(valid_subset, batch_size=1)
        
        # Process data, pair label with mfcc using dictionary
        class_to_mfcc = {i: [] for i in range(10)}
        for batch in train_data:
            feature, label = batch
            scalar_label = label.squeeze(0).argmax().item()
            feature = feature.squeeze(0).numpy().T
            class_to_mfcc[scalar_label].append(feature)
        
        kmeans = fit_kmeans(class_to_mfcc)
        
        # Convert each mfcc to discrete for DHMM
        class_to_discrete = {i: [] for i in range(10)}
        for label, mfcc_list in class_to_mfcc.items():
            for mfcc in mfcc_list:
                discrete = kmeans.predict(mfcc).reshape(-1, 1)
                class_to_discrete[label].append(discrete)
        
        label_to_hmm = fit_dhmm(class_to_discrete)
        acc = 0.0
        for batch in tqdm(valid_data):
            mfcc, label = batch
            mfcc = mfcc.squeeze(0).numpy().T
            label = label.squeeze(0).argmax().item()
            predict_label = classify(kmeans, mfcc, label_to_hmm)
            acc += 1 if predict_label == label else 0
        
        acc_per_fold.append(acc / len(valid_data))
        print(f"Acc: {acc / len(valid_data)}")
    
    print(np.mean(acc_per_fold))

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
    arg_parser.add_argument("--model", type=str, default="hmm")
    arg_parser.add_argument("--kfold", type=int, default=10)
    args = arg_parser.parse_args()
    
    main(args)
