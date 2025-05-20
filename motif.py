
import torch
import torch.nn as nn
import numpy as np
from Bio import SeqIO
import matplotlib.pyplot as plt
import logomaker
import pandas as pd
import os

class CNN(nn.Module):
    def __init__(self, channel=7, nb_filter=16, kernel_size=(4, 10), stride=(1, 1), padding=(0, 0)):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(channel, nb_filter, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(nb_filter),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layer1(x)

    def layer1out(self, x):
        return self.layer1(x).data.cpu().numpy()

def one_hot_encode(seq, max_len=101):
    base_dict = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
    arr = np.zeros((7, 4, max_len))
    for i, base in enumerate(seq):
        if i >= max_len:
            break
        if base.upper() in base_dict:
            arr[0, base_dict[base.upper()], i] = 1
    return arr

def build_pfm(seqs, alphabet='ACGU'):
    pfm = np.zeros((len(seqs[0]), len(alphabet)))
    for seq in seqs:
        for i, base in enumerate(seq):
            if base in alphabet:
                pfm[i, alphabet.index(base)] += 1
    pfm /= np.sum(pfm, axis=1, keepdims=True)
    return pfm

def pfm_to_bits(pfm, background=None):
    if background is None:
        background = np.array([0.25, 0.25, 0.25, 0.25])
    with np.errstate(divide='ignore', invalid='ignore'):
        info = np.log2(pfm / background)
        info[np.isneginf(info)] = 0
        info[np.isnan(info)] = 0
        ic = pfm * info
    return ic

model_path = ".../MultiBranch.pth"
fasta_path = ".../YTHDF1.ls.positives.fa"
output_dir = "cnn_motif_outputs_bits"
os.makedirs(output_dir, exist_ok=True)

num_kernels = 16
top_k = 30
window_size = 10
use_seqs = 100

model = CNN()
state_dict = torch.load(model_path, map_location=torch.device('cpu'))
cnn_state_dict = {k.replace('cnn.', ''): v for k, v in state_dict.items() if k.startswith("cnn.")}
model.load_state_dict(cnn_state_dict, strict=False)
model.eval()

records = list(SeqIO.parse(fasta_path, "fasta"))[:use_seqs]
X = np.array([one_hot_encode(str(rec.seq)) for rec in records])
X_tensor = torch.tensor(X, dtype=torch.float32)

activations = model.layer1out(X_tensor)

for kernel_id in range(num_kernels):
    activation_values = activations[:, kernel_id, 0, :]
    top_indices = np.unravel_index(np.argsort(-activation_values.ravel())[:top_k], activation_values.shape)

    motif_seqs = []
    for sample_idx, pos in zip(*top_indices):
        seq = str(records[sample_idx].seq)
        if pos + window_size <= len(seq):
            motif_seqs.append(seq[pos:pos + window_size])

    if len(motif_seqs) == 0:
        continue

    pfm = build_pfm(motif_seqs)
    bits = pfm_to_bits(pfm)
    df_bits = pd.DataFrame(bits, columns=list("ACGU"))

    plt.figure(figsize=(10, 2))
    logomaker.Logo(df_bits)
    plt.xlabel("Position")
    plt.ylabel("bits")
    plt.ylim([0, 2])
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"cnn_kernel{kernel_id}_motif.png")
    plt.savefig(out_path)
    plt.close()
    print(f"✅ bits logo saved to {out_path}")
