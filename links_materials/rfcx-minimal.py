#!/usr/bin/env python
# coding: utf-8

# My partner will be describing the psuedo labeling generation procedure, performance of different model architectures and
# loss functions in more detail. Here are some points that i found to be important.
# 1. Catastrophic forgetting in neural networks
# There is imbalance in the distribution of bird species, for ex: species 3 occurs very frequently.
# During training I found that initially model learns to classify species 3 and as the training proceeds
# it starts "forgetting". The confidence for species 3 goes on decreasing which negatively impacts the lb score.
# So, we need to make sure that other species are learnt without forgetting species 3. 
# I found that recall rate for species 3 can be improved by setting pos_weight in BCELoss.
# You may find this paper interesting if you are more curious: https://arxiv.org/pdf/1612.00796.pdf (especially section 2.1)
# 2. Augmenting other datasets\
# Not all parts of the audio are occupied by bird species. I replaced these unoccupied parts with bird
# songs from cornell. 
# 3. Misc
#     - Validation scheme should be similar to test scheme.
#     For ex: If you feed 5s chunks during test and then take max, the same thing should
#     be done during validation also.
#     - I found Click Noise Augmentation to be very useful (https://librosa.org/doc/0.8.0/generated/librosa.clicks.html)
#     - Using pretrained weights (imagenet/cornell) can help to converge much faster.
#     - Model Averaging seems to always lead to better generalization.
#     - 5s crops seems to perform slightly better than 10s crops

# In[1]:


get_ipython().system('pip install resnest > /dev/null')
get_ipython().system('pip install colorednoise > /dev/null')


# In[2]:


import albumentations as A
from resnest.torch.resnet import ResNet, Bottleneck
import random
from glob import glob
from collections import OrderedDict
import os.path as osp
import os
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from skimage.transform import resize
from torchvision.models import resnet18, resnet34, resnet50
from resnest.torch import resnest50
from tqdm.auto import tqdm
import colorednoise as cn
import librosa
import torchaudio
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')


# In[3]:


def seed_everything(seed=42):
    print(f'setting everything to seed {seed}')
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.empty_cache()
    
seed_everything(42)


# In[4]:


# https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/198418
# label-level average
# Assume float preds [BxC], labels [BxC] of 0 or 1
def LWLRAP(preds, labels):
    # Ranks of the predictions
    ranked_classes = torch.argsort(preds, dim=-1, descending=True)
    # i, j corresponds to rank of prediction in row i
    class_ranks = torch.zeros_like(ranked_classes).to(preds.device)
    for i in range(ranked_classes.size(0)):
        for j in range(ranked_classes.size(1)):
            class_ranks[i, ranked_classes[i][j]] = j + 1
    # Mask out to only use the ranks of relevant GT labels
    ground_truth_ranks = class_ranks * labels + (1e6) * (1 - labels)
    # All the GT ranks are in front now
    sorted_ground_truth_ranks, _ = torch.sort(
        ground_truth_ranks, dim=-1, descending=False)
    # Number of GT labels per instance
    num_labels = labels.sum(-1)
    pos_matrix = torch.tensor(
        np.array([i+1 for i in range(labels.size(-1))])).unsqueeze(0).to(preds.device)
    score_matrix = pos_matrix / sorted_ground_truth_ranks
    score_mask_matrix, _ = torch.sort(labels, dim=-1, descending=True)
    scores = score_matrix * score_mask_matrix
    score = scores.sum() / labels.sum()
    return score.item()


# In[5]:


class Config:
    batch_size = 8
    weight_decay = 1e-8
    lr = 1e-3
    num_workers = 4
    epochs = 6
    num_classes = 24
    sr = 32_000
    duration = 5
    total_duration = 60
    nmels = 128
    EXTRAS_DIR = "../input/rfcxextras"
    ROOT = "../input/rfcx-species-audio-detection"
    TRAIN_AUDIO_ROOT = osp.join(ROOT, "train")
    TEST_AUDIO_ROOT = osp.join(ROOT, "test")
    loss_fn = torch.nn.BCEWithLogitsLoss()


# # Audio Augmentations

# In[6]:


# Mostly taken from https://www.kaggle.com/hidehisaarai1213/rfcx-audio-data-augmentation-japanese-english
class AudioTransform:
    def __init__(self, always_apply=False, p=0.5):
        self.always_apply = always_apply
        self.p = p

    def __call__(self, y: np.ndarray):
        if self.always_apply:
            return self.apply(y)
        else:
            if np.random.rand() < self.p:
                return self.apply(y)
            else:
                return y

    def apply(self, y: np.ndarray):
        raise NotImplementedError


class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, y: np.ndarray):
        for trns in self.transforms:
            y = trns(y)
        return y


class OneOf:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, y: np.ndarray):
        n_trns = len(self.transforms)
        trns_idx = np.random.choice(n_trns)
        trns = self.transforms[trns_idx]
        return trns(y)


class GaussianNoiseSNR(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, min_snr=5.0, max_snr=20.0, **kwargs):
        super().__init__(always_apply, p)

        self.min_snr = min_snr
        self.max_snr = max_snr

    def apply(self, y: np.ndarray, **params):
        snr = np.random.uniform(self.min_snr, self.max_snr)
        a_signal = np.sqrt(y ** 2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        white_noise = np.random.randn(len(y))
        a_white = np.sqrt(white_noise ** 2).max()
        augmented = (y + white_noise * 1 / a_white * a_noise).astype(y.dtype)
        return augmented


class PinkNoiseSNR(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, min_snr=5.0, max_snr=20.0, **kwargs):
        super().__init__(always_apply, p)

        self.min_snr = min_snr
        self.max_snr = max_snr

    def apply(self, y: np.ndarray, **params):
        snr = np.random.uniform(self.min_snr, self.max_snr)
        a_signal = np.sqrt(y ** 2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        pink_noise = cn.powerlaw_psd_gaussian(1, len(y))
        a_pink = np.sqrt(pink_noise ** 2).max()
        augmented = (y + pink_noise * 1 / a_pink * a_noise).astype(y.dtype)
        return augmented


class TimeShift(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, max_shift_second=2, sr=32000, padding_mode="zero"):
        super().__init__(always_apply, p)

        assert padding_mode in [
            "replace", "zero"], "`padding_mode` must be either 'replace' or 'zero'"
        self.max_shift_second = max_shift_second
        self.sr = sr
        self.padding_mode = padding_mode

    def apply(self, y: np.ndarray, **params):
        shift = np.random.randint(-self.sr * self.max_shift_second,
                                  self.sr * self.max_shift_second)
        augmented = np.roll(y, shift)
        # if self.padding_mode == "zero":
        #     if shift > 0:
        #         augmented[:shift] = 0
        #     else:
        #         augmented[shift:] = 0
        return augmented


class VolumeControl(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, db_limit=10, mode="uniform"):
        super().__init__(always_apply, p)

        assert mode in ["uniform", "fade", "fade", "cosine", "sine"],             "`mode` must be one of 'uniform', 'fade', 'cosine', 'sine'"

        self.db_limit = db_limit
        self.mode = mode

    def apply(self, y: np.ndarray, **params):
        db = np.random.uniform(-self.db_limit, self.db_limit)
        if self.mode == "uniform":
            db_translated = 10 ** (db / 20)
        elif self.mode == "fade":
            lin = np.arange(len(y))[::-1] / (len(y) - 1)
            db_translated = 10 ** (db * lin / 20)
        elif self.mode == "cosine":
            cosine = np.cos(np.arange(len(y)) / len(y) * np.pi * 2)
            db_translated = 10 ** (db * cosine / 20)
        else:
            sine = np.sin(np.arange(len(y)) / len(y) * np.pi * 2)
            db_translated = 10 ** (db * sine / 20)
        augmented = y * db_translated
        return augmented


# In[7]:


def mono_to_color(X, eps=1e-6, mean=None, std=None):
    X = np.stack([X, X, X], axis=-1)

    # Standardize
    mean = mean or X.mean()
    std = std or X.std()
    X = (X - mean) / (std + eps)

    # Normalize to [0, 255]
    _min, _max = X.min(), X.max()

    if (_max - _min) > eps:
        V = np.clip(X, _min, _max)
        V = 255 * (V - _min) / (_max - _min)
        V = V.astype(np.uint8)
    else:
        V = np.zeros_like(X, dtype=np.uint8)

    return V


def normalize(image, mean=None, std=None):
    image = image / 255.0
    if mean is not None and std is not None:
        image = (image - mean) / std
    return image.astype(np.float32)


# In[8]:


class RFCDataset:
    def __init__(self, tp, fp=None, config=None,
                 mode='train', inv_counts=None):
        self.tp = tp
        self.fp = pd.read_csv("../input/rfcxextras/cornell-train.csv")
        self.fp = self.fp[self.fp.ebird_code<'c'].reset_index(drop=True)
        self.fp_root = "../input/birdsong-resampled-train-audio-00/"        
        self.inv_counts = inv_counts
        self.config = config
        self.sr = self.config.sr
        self.total_duration = self.config.total_duration
        self.duration = self.config.duration
        self.data_root = self.config.TRAIN_AUDIO_ROOT
        self.nmels = self.config.nmels
        self.fmin, self.fmax = 84, self.sr//2
        self.mode = mode
        self.num_classes = self.config.num_classes
        self.resampler = torchaudio.transforms.Resample(
            orig_freq=48_000, new_freq=self.sr)
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=self.sr, n_mels=self.nmels,
                                                        f_min=self.fmin, f_max=self.fmax,
                                                        n_fft=2048)
        self.transform = Compose([
            OneOf([
                GaussianNoiseSNR(min_snr=10),
                PinkNoiseSNR(min_snr=10)
            ]),
            TimeShift(sr=self.sr),
            VolumeControl(p=0.5)
        ])
        self.img_transform = A.Compose([
            A.OneOf([
                A.Cutout(max_h_size=5, max_w_size=20),
                A.CoarseDropout(max_holes=4),
                A.RandomBrightness(p=0.25),
            ], p=0.5)])
        self.num_splits = self.config.total_duration//self.duration
        assert self.config.total_duration == self.duration *             self.num_splits, "not a multiple"

    def __len__(self):
        return len(self.tp)

    def __getitem__(self, idx):
        labels = np.zeros((self.num_classes,), dtype=np.float32)

        recording_id = self.tp.loc[idx, 'recording_id']
        df = self.tp.loc[self.tp.recording_id == recording_id]
        maybe_labels = df.species_id.unique()
        np.put(labels, maybe_labels, 0.2)

        df = df.sample(weights=df.species_id.apply(
            lambda x: self.inv_counts[x]))
        fn = osp.join(self.data_root, f"{recording_id}.flac")
        df = df.squeeze()
        t0 = max(df['t_min'], 0)
        t1 = max(df['t_max'], 0)
        t0 = np.random.uniform(t0, t1)
        t0 = max(t0, 0)
        t0 = min(t0, self.total_duration-self.duration)
        t1 = t0 + self.duration
        valid_df = self.tp[self.tp.recording_id == recording_id]
        valid_df = valid_df[(valid_df.t_min < t1) & (valid_df.t_max > t0)]
        y, _ = librosa.load(fn, sr=None, offset=t0,
                            duration=self.duration)
        if len(valid_df):
            np.put(labels, valid_df.species_id.unique(), 1)
        np.put(labels, df.species_id, 1)

        if random.random()<0.5:
            end_idx = int((valid_df.t_max.max() - t0)*self.sr)
            rem_len = max(0, len(y) - end_idx)
            idx = np.random.randint(0, len(self.fp))
            
            fn = osp.join(self.fp_root, self.fp.ebird_code[idx],self.fp.filename[idx])
            fn = fn.replace('mp3', 'wav')
            y_other, _ = librosa.load(fn, sr=self.sr,
                                    duration=None, mono=True,
                                    res_type='kaiser_fast')
            aug_len = min(len(y_other), rem_len)
            y[end_idx:end_idx+aug_len] = y_other[:aug_len]

        y = self.resampler(torch.from_numpy(y).float()).numpy()
        # do augmentation
        y = self.transform(y)
        if random.random() < 0.25:
            tempo, beats = librosa.beat.beat_track(y=y, sr=self.sr)
            y = librosa.clicks(frames=beats, sr=self.sr, length=len(y))

        melspec = librosa.feature.melspectrogram(
            y, sr=self.sr, n_mels=self.nmels, fmin=self.fmin, fmax=self.fmax,
        )
        melspec = librosa.power_to_db(melspec)
        melspec = mono_to_color(melspec)
        melspec = normalize(melspec, mean=None, std=None)
        melspec = self.img_transform(image=melspec)['image']
        melspec = np.moveaxis(melspec, 2, 0)
        return melspec, labels


# In[9]:


class RFCTestDataset:
    def __init__(self, tp, fp=None, config=None,
                 mode='test'):
        self.tp = tp
        self.fp = fp
        self.config = config
        self.sr = self.config.sr
        self.duration = self.config.duration
        if mode == 'val':
            self.data_root = self.config.TRAIN_AUDIO_ROOT
        else:
            self.data_root = self.config.TEST_AUDIO_ROOT

        self.nmels = self.config.nmels
        self.fmin, self.fmax = 84, self.sr//2
        self.mode = mode
        self.resampler = torchaudio.transforms.Resample(
            orig_freq=48_000, new_freq=self.sr)
        self.num_classes = self.config.num_classes
        self.num_splits = self.config.total_duration//self.duration
        assert self.config.total_duration == self.duration *             self.num_splits, "not a multiple"

    def __len__(self):
        return len(self.tp.recording_id.unique())

    def __getitem__(self, idx):
        recording_id = self.tp.loc[idx, 'recording_id']
        df = self.tp.loc[self.tp.recording_id == recording_id]
        if self.mode == 'val':
            fn = f"{self.config.EXTRAS_DIR}/train_melspec32k_10s/train_melspec32k_10s/{recording_id}.npy"
        else:
            fn = f"{self.config.EXTRAS_DIR}/test_melspec32k_10s/test_melspec32k_10s/{recording_id}.npy"
        try:
            melspec_stacked = np.load(fn)
        except:
            audio_fn = osp.join(self.data_root, f"{recording_id}.flac")
            y, _ = librosa.load(audio_fn, sr=None,
                                duration=self.config.total_duration)
            # split into n arrays
            y_stacked = np.stack(np.split(y, self.num_splits), 0)
            melspec_stacked = []
            for y in y_stacked:
                y = self.resampler(torch.from_numpy(y).float()).numpy()
                melspec = librosa.feature.melspectrogram(
                    y, sr=self.sr, n_mels=self.nmels, fmin=self.fmin, fmax=self.fmax,
                )
                melspec = librosa.power_to_db(melspec)
                melspec = mono_to_color(melspec)
                melspec = normalize(melspec, mean=None, std=None)
                melspec = np.moveaxis(melspec, 2, 0)
                melspec_stacked.append(melspec)

            melspec_stacked = np.stack(melspec_stacked)
            np.save(fn, melspec_stacked)

        if self.mode == 'val':
            species = df.loc[:, 'species_id'].unique()
            labels = np.zeros((self.num_classes,))
            np.put(labels, species, 1)

            return melspec_stacked, labels
        else:
            melspec_stacked = np.load(fn)
            return melspec_stacked


# In[10]:


# resnest 50 trained on cornell 
# https://www.kaggle.com/theoviel/birds-cp-1
MODEL_CONFIGS = {
    "resnest50_fast_1s1x64d":
    {
        "num_classes": 264,
        "block": Bottleneck,
        "layers": [3, 4, 6, 3],
        "radix": 1,
        "groups": 1,
        "bottleneck_width": 64,
        "deep_stem": True,
        "stem_width": 32,
        "avg_down": True,
        "avd": True,
        "avd_first": True
    }
}


def get_model(pretrained=True, n_class=24):
    # model = torchvision.models.resnext50_32x4d(pretrained=False)
    # model = torchvision.models.resnext101_32x8d(pretrained=False)
    model = ResNet(**MODEL_CONFIGS["resnest50_fast_1s1x64d"])
    n_features = model.fc.in_features
    model.fc = nn.Linear(n_features, 264)
    # model.load_state_dict(torch.load('resnext50_32x4d_extra_2.pt'))
    # model.load_state_dict(torch.load('resnext101_32x8d_wsl_extra_4.pt'))
    fn = '../input/birds-cp-1/resnest50_fast_1s1x64d_conf_1.pt'
    model.load_state_dict(torch.load(fn, map_location='cpu'))
    model.fc = nn.Linear(n_features, n_class)
    return model


# In[11]:


class BaseNet(LightningModule):
    def __init__(self, config, train_recid, val_recid):
        super().__init__()
        self.config = config
        self.batch_size = self.config.batch_size
        self.num_workers = self.config.num_workers
        self.lr = self.config.lr
        self.epochs = self.config.epochs

        self.weight_decay = self.config.weight_decay
        # to improve species 3 recall rate
        pos_weight = torch.ones((24,))
        pos_weight[3] = 4
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.sr = self.config.sr
        self.train_recid = train_recid
        self.val_recid = val_recid

    def train_dataloader(self):
        tp = train_tp[train_tp.recording_id.isin(
            self.train_recid)].reset_index(drop=True)
        self.train_recid = tp.recording_id.unique()
        inv_counts = dict(1/tp.species_id.value_counts())
        weights = tp.species_id.apply(lambda x: inv_counts[x])
        tp_aug = new_labels[new_labels.recording_id.isin(tp.recording_id)]
        tp = pd.concat([tp, tp_aug], ignore_index=True)
        train_dataset = RFCDataset(tp, train_fp,
                                   config=self.config,
                                   mode='train',
                                   inv_counts=inv_counts)
        train_sampler = WeightedRandomSampler(weights, num_samples=len(train_dataset),
                                              replacement=True)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                  num_workers=self.num_workers,
                                  sampler=train_sampler,
                                  drop_last=True,
                                  pin_memory=True)
        return train_loader

    def val_dataloader(self):
        val_tp = train_tp[train_tp.recording_id.isin(
            self.val_recid)].reset_index(drop=True)
        val_recid = val_tp.recording_id.unique()
        overlap = set(val_recid).intersection(set(self.train_recid))
#         print('overlapped ids', overlap)
        val_tp = val_tp[~val_tp.recording_id.isin(overlap)]
        val_tp_aug = new_labels[new_labels.recording_id.isin(
            val_tp.recording_id)]
        val_tp = pd.concat([val_tp, val_tp_aug], ignore_index=True)
        val_dataset = RFCTestDataset(val_tp, train_fp,
                                     config=self.config,
                                     mode='val')
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size,
                                num_workers=self.num_workers, shuffle=False,
                                pin_memory=True)
        return val_loader

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=self.config.lr,
                                  weight_decay=self.config.weight_decay)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optim,
                                                                    mode='min',
                                                                    factor=0.5,
                                                                    patience=2,
                                                                    verbose=True),
            'monitor': 'val_loss',
            'interval': 'epoch',
            'frequency': 1,
            'strict': True,
        }

        self.optimizer = optim
        self.scheduler = scheduler

        return [optim], [scheduler]


# In[12]:


class RFCNet(BaseNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        n_class = self.config.num_classes
        self.model = get_model(
            pretrained=True, n_class=n_class)
        self.cnf_matrix = np.zeros((n_class, n_class))

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        with torch.no_grad():
            lwlrap = LWLRAP(preds, y)
        metrics = {"train_loss": loss.item(), "train_lwlrap": lwlrap}
        self.log_dict(metrics,
                      on_epoch=True, on_step=True)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x, y = batch
        for i, x_partial in enumerate(torch.split(x, 1, dim=1)):
            x_partial = x_partial.squeeze(1)
            if i == 0:
                preds = self(x_partial)
            else:
                # take max over predictions
                preds = torch.max(preds, self(x_partial))
        val_loss = self.loss_fn(preds, y).item()
        val_lwlrap = LWLRAP(preds, y)
        # loss is tensor. The Checkpoint Callback is monitoring 'checkpoint_on'
        metrics = {"val_loss": val_loss, "val_lwlrap": val_lwlrap}
        self.log_dict(metrics, prog_bar=True,
                      on_epoch=True, on_step=True)


# # Average model weights

# In[13]:


def average_model(paths):
    weights = np.ones((len(paths),))
    weights = weights/weights.sum()
    for i, p in enumerate(paths):
        m = torch.load(p)['state_dict']
        if i == 0:
            averaged_w = OrderedDict()
            for k in m.keys():
                if 'pos' in k: continue
                # remove pl prefix in state dict
                knew = k.replace('model.', '')
                averaged_w[knew] = weights[i]*m[k]
        else:
            for k in m.keys():
                if 'pos' in k: continue
                knew = k.replace('model.', '')
                averaged_w[knew] = averaged_w[knew] + weights[i]*m[k]
    return averaged_w


# # Model training

# In[14]:


config = Config()
train_tp = pd.read_csv(osp.join(config.ROOT, 'train_tp.csv'))

fold_df = pd.read_csv(
    osp.join(config.EXTRAS_DIR, 'preprocessed_rainforest_dataset.csv'))
fn = "../input/extra-labels-for-rcfx-competition-data/extra_labels_v71.csv"
print(fn)
new_labels = pd.read_csv(fn)
new_labels['t_diff'] = new_labels['t_max'] - new_labels['t_min']
idx = np.where(new_labels['t_diff'] < 0)[0]
new_labels = new_labels.drop(idx, axis=0).reset_index(drop=True)
num_folds = len(fold_df.fold.unique())
train_fp = pd.read_csv(osp.join(config.ROOT, 'train_fp.csv'))
for fold in range(num_folds):
    print('\n\nTraining fold', fold)
    print('*' * 40)

    train_recid = fold_df[fold_df.fold != fold].recording_id
    val_recid = fold_df[fold_df.fold == fold].recording_id
    model = RFCNet(config=config, train_recid=train_recid,
                   val_recid=val_recid)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_lwlrap_epoch',
        filename='{epoch:02d}-{val_loss_epoch:.2f}-{val_lwlrap_epoch:.2}',
        mode='max',
        save_top_k=5,
        save_weights_only=True,
    )
    early_stopping = EarlyStopping(monitor='val_lwlrap_epoch', mode='max', patience=5,
                                   verbose=True)
    trainer = Trainer(gpus=1,
                      max_epochs=config.epochs,
                      progress_bar_refresh_rate=1,                      
                      #   gradient_clip_val=2,
                      accumulate_grad_batches=4,
                      num_sanity_val_steps=0,
                      callbacks=[checkpoint_callback, early_stopping])

    trainer.fit(model)


# # Model Validation

# In[15]:


def get_one_hot(targets, nb_classes=24):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])


sub = pd.read_csv(osp.join(config.ROOT, 'sample_submission.csv'))
species_cols = list(sub.columns)
species_cols.remove('recording_id')

cv_preds = pd.DataFrame(columns=species_cols)
cv_preds['recording_id'] = train_tp['recording_id'].drop_duplicates()
cv_preds = cv_preds.set_index('recording_id')

label_df = pd.DataFrame(columns=species_cols)
label_df['recording_id'] = train_tp['recording_id'].drop_duplicates()
label_df = label_df.set_index('recording_id')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_model(pretrained=False)
model.to(device)
for fold in range(num_folds):
    paths = glob(f"./lightning_logs/version_{fold}/checkpoints/*.ckpt")
    print(paths)
    averaged_w = average_model(paths)
    model.load_state_dict(averaged_w)
    model.eval()
    train_recid = fold_df[fold_df.fold!=fold].recording_id
    val_recid = fold_df[fold_df.fold==fold].recording_id

    val_tp = train_tp[train_tp.recording_id.isin(val_recid)].reset_index(drop=True)
    val_recid = val_tp.recording_id.unique()
    overlap = set(val_recid).intersection(set(train_recid))
    val_tp = val_tp[~val_tp.recording_id.isin(overlap)]
    val_tp_aug = new_labels[new_labels.recording_id.isin(val_tp.recording_id)]
    val_tp = pd.concat([val_tp, val_tp_aug], ignore_index=True)

    dataset = RFCTestDataset(val_tp, config=config, mode='val')
    test_loader = DataLoader(dataset, batch_size=config.batch_size,
                             num_workers=config.num_workers,
                             shuffle=False, drop_last=False)
    tk = test_loader
    with torch.no_grad():
        fold_preds, labels = [], []
        for i, (im, l) in enumerate(tk):
            # continue
            im = im.to(device)
            for j, x_partial in enumerate(torch.split(im, 1, dim=1)):
                x_partial = x_partial.squeeze(1)
                if j == 0:
                    preds = model(x_partial)
                else:
                    preds = torch.max(preds, model(x_partial))


            o = preds.sigmoid().cpu().numpy()
            # o = preds.cpu().numpy()
            fold_preds.extend(o)
            labels.extend(l.cpu().numpy())
        # continue
        p = torch.from_numpy(np.array(fold_preds)) 
        t = torch.from_numpy(np.array(labels))
        print(f"lwlrap: {LWLRAP(p, t):.6}")
        cv_preds.loc[val_recid, species_cols] = fold_preds
        label_df.loc[val_recid, species_cols] = labels

# print(cv_preds.head())
cv_preds.to_csv('cv_preds.csv')

recid = train_tp['recording_id'].values
cv_preds = cv_preds.loc[recid].values.astype(np.float32)
cv_preds = torch.from_numpy(cv_preds)

labels = label_df.loc[recid].values.astype(np.float32)
labels = torch.from_numpy(labels)

print(f"lwlrap: {LWLRAP(cv_preds, labels):.6}")


# # Test predictions

# In[16]:


sub = pd.read_csv(osp.join(config.ROOT, 'sample_submission.csv'))
species_cols = list(sub.columns)
species_cols.remove('recording_id')
# initialize to zero.
sub.loc[:, species_cols] = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_model(pretrained=False)
model.to(device)
for fold in range(num_folds):
    paths = glob(f"./lightning_logs/version_{fold}/checkpoints/*.ckpt")
    print(paths)
    averaged_w = average_model(paths)
    model.load_state_dict(averaged_w)
    model.eval()
    dataset = RFCTestDataset(sub, config=config, mode='test')
    test_loader = DataLoader(dataset, batch_size=config.batch_size,
                             num_workers=4,
                             shuffle=False, drop_last=False)
    tk = tqdm(test_loader, total=len(test_loader))
    sub_index = 0
    with torch.no_grad():
        for i, im in enumerate(tk):
            im = im.to(device)
            for i, x_partial in enumerate(torch.split(im, 1, dim=1)):
                x_partial = x_partial.squeeze(1)
                if i == 0:
                    preds = model(x_partial)
                else:
                    # take max over predictions
                    preds = torch.max(preds, model(x_partial))

            o = preds.sigmoid().cpu().numpy()
            # o = preds.cpu().numpy()
            for val in o:
                sub.loc[sub_index, species_cols] += val
                sub_index += 1

# # take average of predictions
sub.loc[:, species_cols] /= num_folds
sub.to_csv('submission.csv', index=False)
print(sub.head())
print(sub.max(1).head())


# In[17]:


sub.iloc[:, 1:].describe()

