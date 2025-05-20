# RMDNet： RNA-aware Attention-based Multi-Branch Integration Network
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(BASE_DIR))
import numpy as np
np.set_printoptions(threshold=np.inf)
import pdb
import torch    
import torch.nn as nn
import subprocess
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv, dense_diff_pool
from torch_geometric.utils import to_dense_batch, to_dense_adj
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix
)
import argparse
import pickle
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# GPU or CPU
if torch.cuda.is_available():
        cuda = True
        # torch.cuda.set_device(1)
        torch.cuda.set_device(0)
        print('===> Using GPU')
else:
        cuda = False
        print('===> Using CPU')

def evaluate_all_metrics(y_true, y_score, output_path="results.txt", title="Model Evaluation"):
    y_pred = (y_score > 0.5).astype(int)
    auc = roc_auc_score(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp + 1e-8)
    else:
        specificity = 0.0

    with open(output_path, 'a') as f:
        f.write(f"========== {title} ==========\n")
        f.write(f"AUC:          {auc:.4f}\n")
        f.write(f"PR-AUC:       {pr_auc:.4f}\n")
        f.write(f"Accuracy:     {acc:.4f}\n")
        f.write(f"Precision:    {prec:.4f}\n")
        f.write(f"Recall:       {rec:.4f}\n")
        f.write(f"Specificity:  {specificity:.4f}\n")
        f.write(f"F1 Score:     {f1:.4f}\n")
        f.write(f"MCC:          {mcc:.4f}\n")
        f.write("="*40 + "\n")

# Filling subsequences with base nucleotide N
def padding_sequence(seq, max_len=501, repkey='N'):
    seq_len = len(seq)
    if seq_len < max_len:
        return seq + repkey * (max_len - seq_len)
    return seq[:max_len]
# Convert the sequence into a one-hot coding matrix

def get_RNA_seq_concolutional_array(seq, motif_len = 4):
    seq = seq.replace('U', 'T')
    alpha = 'ACGT'
    row = (len(seq) + 2*motif_len - 2)
    new_array = np.zeros((row, 4))
    for i in range(motif_len-1):
        new_array[i] = np.array([0.25]*4)
    
    for i in range(row-3, row):
        new_array[i] = np.array([0.25]*4)
        
    for i, val in enumerate(seq):
        i = i + motif_len-1
        if val not in 'ACGT':
            new_array[i] = np.array([0.25]*4)
            continue
        try:
            index = alpha.index(val)
            new_array[i][index] = 1
        except:
            pdb.set_trace()

    return new_array

# Divide RNA sequence into multiple sub sequences with partial overlap

def predict_struct_rnafold(seq):

    print("RNAfold predictions are being run：", seq[:20])
    try:
        result = subprocess.run(['RNAfold'], input=seq.encode(), capture_output=True, check=True)
        lines = result.stdout.decode().split('\n')
        struct_line = lines[1]
        structure = struct_line.strip().split(' ')[0]  # 提取 "(((...)))"
        mfe_str = struct_line.strip().split(' ')[-1]  # (-7.80)
        mfe = float(mfe_str.replace('(', '').replace(')', ''))
        pair_probs = [1.0 if c in '()' else 0.0 for c in structure]
        return structure, pair_probs, mfe
    except Exception as e:
        print(f"RNAfold prediction error: {e}")
        fallback = '.' * len(seq)
        return fallback, [0.0] * len(seq), 0.0

def load_struct_graphs(seq_list, use_rnafold=True, dataset_name="default", max_workers=8):

    cache_path = f"{dataset_name}_struct.pkl"
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            struct_data_list = pickle.load(f)
            print(f"Loaded Structure Chart Cache. {cache_path}， {len(struct_data_list)} ")
            return struct_data_list

    print(f"RNA structure maps are being constructed in parallel (first run):{dataset_name}， {len(seq_list)} ")

    def worker(seq):
        try:
            struct, pair_probs, mfe = predict_struct_rnafold(seq)
            graph = seq_to_graph(seq, struct, pair_probs, mfe)
            return graph
        except Exception as e:
            print(f"[Skip] Failed to generate a subsequence structure diagram. {seq[:10]}... → {e}")
            return None

    struct_data_list = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker, seq) for seq in seq_list]
        for future in tqdm(as_completed(futures), total=len(futures), desc="build an architecture diagram"):
            result = future.result()
            struct_data_list.append(result)


    failed_count = sum([g is None for g in struct_data_list])
    if failed_count > 0:
        print(f"Found {failed_count} structural graph builds failing, using empty graph placeholder")

        dummy_graph = Data(x=torch.zeros((1, 6)), edge_index=torch.empty((2, 0), dtype=torch.long))

        struct_data_list = [
            g if g is not None else dummy_graph for g in struct_data_list
        ]

    print(f"Number of final structural drawings: {len(struct_data_list)}")

    with open(cache_path, "wb") as f:
        pickle.dump(struct_data_list, f)
        print(f"The build is complete and the structural map cache has been saved: {cache_path}")
    return struct_data_list

def split_overlap_seq(seq, window_size):
    overlap_size = 50
    bag_seqs = []
    seq_len = len(seq)
    remain_ins = 0
    if seq_len >= window_size:
        num_ins = (seq_len - window_size) // (window_size - overlap_size) + 1
        remain_ins = (seq_len - window_size)%(window_size - overlap_size)
    else:
        num_ins = 0
    end = 0
    for ind in range(num_ins):
        start = max(end - overlap_size, 0)
        end = start + window_size
        subseq = seq[start:end]
        bag_seqs.append(subseq)
    if remain_ins > 10:
        bag_seqs.append(padding_sequence(seq[-window_size:], window_size))
    elif num_ins == 0:
        bag_seqs.append(padding_sequence(seq, window_size))
    return bag_seqs
#该函数的目的是将RNA序列分割成多个具有部分重叠的子序列，并对需要填充的子序列进行填充处理。

def seq_to_graph(seq, struct=None, pair_probs=None):
    if struct is None:
        struct = '.' * len(seq)
    if pair_probs is None:
        pair_probs = [0.0] * len(seq)

    x = []
    edge_index = [[], []]
    for i, base in enumerate(seq):
        onehot = [0] * 4
        if base in "ACGU":
            onehot["ACGU".index(base)] = 1
        paired = 0 if struct[i] == '.' else 1
        prob = pair_probs[i] if pair_probs else 0.0
        x.append(onehot + [paired, prob])
    x = torch.tensor(x, dtype=torch.float)

    for i in range(len(seq) - 1):
        edge_index[0] += [i, i+1]
        edge_index[1] += [i+1, i]
    stack = []
    for i, s in enumerate(struct):
        if s == '(':
            stack.append(i)
        elif s == ')':
            j = stack.pop()
            edge_index[0] += [i, j]
            edge_index[1] += [j, i]
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    return Data(x=x, edge_index=edge_index)

# Read sequence file
def read_seq_graphprot(seq_file, label = 1):
    seq_list = []
    labels = []
    with open(seq_file, 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name = line[1:-1]
            else:
                seq = line[:-1].upper()
                seq = seq.replace('T', 'U')
                seq_list.append(seq)
                labels.append(label)
    return seq_list, labels

# Obtain the processed one-hot coding matrix and corresponding labels 获取处理后的一热编码矩阵和对应的标签

def get_bag_data(data, channel = 7, window_size = 101):
    bags = []
    seqs = data["seq"]
    labels = data["Y"]
    for seq in seqs:
        bag_seqs = split_overlap_seq(seq, window_size = window_size)
        bag_subt = []
        for bag_seq in bag_seqs:
            tri_fea = get_RNA_seq_concolutional_array(bag_seq)
            bag_subt.append(tri_fea.T)
        num_of_ins = len(bag_subt)
        if num_of_ins > channel:
            start = (num_of_ins - channel)/2
            bag_subt = bag_subt[start: start + channel]
        if len(bag_subt) <channel:
            rand_more = channel - len(bag_subt)
            for ind in range(rand_more):
                tri_fea = get_RNA_seq_concolutional_array('N'*window_size)
                bag_subt.append(tri_fea.T)
        bags.append(np.array(bag_subt))
    return bags, labels

def get_bag_data_1_channel(data, max_len = 501):
    bags = []
    seqs = data["seq"]
    labels = data["Y"]
    for seq in seqs:
        bag_seq = padding_sequence(seq, max_len = max_len)
        bag_subt = []
        tri_fea = get_RNA_seq_concolutional_array(bag_seq)
        bag_subt.append(tri_fea.T)
        bags.append(np.array(bag_subt))
    return bags, labels

# Gather positive and negative data together
def read_data_file(posifile, negafile = None, train = True):
    data = dict()
    seqs, labels = read_seq_graphprot(posifile, label = 1)
    if negafile:
        seqs2, labels2 = read_seq_graphprot(negafile, label = 0)
        seqs = seqs + seqs2
        labels = labels + labels2
        
    data["seq"] = seqs
    data["Y"] = np.array(labels)
    
    return data

def get_data(posi, nega, channel,  window_size, train = True): #
    data = read_data_file(posi, nega, train = train)
    if channel == 1:
        train_bags, label = get_bag_data_1_channel(data, max_len = window_size)

    else:
        train_bags, label = get_bag_data(data, channel = channel, window_size = window_size)

    return train_bags, label


class EarlyStopper:
    def __init__(self, patience=2, min_delta=0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class Estimator(object):

    def __init__(self, model, model_file=None):
        self.model = model
        self.model_file = model_file

    def compile(self, optimizer, loss):
        self.optimizer = optimizer
        self.loss_f = loss
        self.criterion =  nn.CrossEntropyLoss()

    def _fit(self, train_loader):
        """
        train one epoch
        """
        loss_list = []
        for idx, (X, y) in enumerate(train_loader):
            X_v = Variable(X)
            y_v = Variable(y)
            # print np.array(X_v).shape
            if cuda:
                X_v = X_v.cuda()
                y_v = y_v.cuda()
            self.optimizer.zero_grad()
            y_pred = self.model(X_v)
            loss = self.loss_f(y_pred, y_v)
            loss.backward()
            self.optimizer.step()

            ## for log
            loss_list.append(loss.item())

        return sum(loss_list) / len(loss_list)

    def _fit_struct(self, data_iter):
        total_loss = 0
        self.model.train()
        for (seq_x, labels), struct_data in data_iter:
            seq_x, labels = seq_x.cuda(), labels.cuda()
            struct_data = struct_data.cuda()

            self.optimizer.zero_grad()
            outputs = self.model(seq_x, struct_data)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss

    def fit(self, X, y, batch_size=32, nb_epoch=10, validation_data=(), struct_data=None, struct_val_data=None):
        print(X.shape)

        if struct_data is not None:
            train_dataset = RNA_Dataset(torch.from_numpy(X.astype(np.float32)),
                                        y.astype(np.float32).astype(int),
                                        struct_data)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
            use_struct = True
        else:
            X_tensor = torch.from_numpy(X.astype(np.float32))
            y_tensor = torch.from_numpy(y.astype(np.float32)).long().view(-1)
            train_set = TensorDataset(X_tensor, y_tensor)
            train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
            use_struct = False

        early_stopper = EarlyStopper(patience=20, min_delta=0.0005)
        min_epoch = 15
        train_losses = []
        best_auc = 0.0
        best_epoch = 0

        for t in range(nb_epoch):
            self.model.train()
            if use_struct:
                epoch_loss = self._fit_struct(train_loader)
            else:
                epoch_loss = self._fit(train_loader)

            train_losses.append(epoch_loss)
            print(f"Epoch {t + 1}/{nb_epoch} loss: {epoch_loss:.4f}")

            if validation_data:
                X_val, y_val = validation_data
                val_loss, val_auc = self.evaluate(X_val, y_val, struct_data=struct_val_data)

                if val_auc > best_auc:
                    best_auc = val_auc
                    best_epoch = t + 1
                    model_path = f"{self.model_file}.best_auc.pth"
                    torch.save(self.model.state_dict(), model_path)

                if t + 1 >= min_epoch and early_stopper(-val_auc):
                    print(f"AUC-based Early stopping at epoch {t + 1}")
                    break

        final_loss = train_losses[-1]
        print(f"Best AUC: {best_auc:.4f} at epoch {best_epoch}")
        return final_loss, best_auc

    def evaluate(self, X, y, struct_data=None, batch_size=256):
        self.model.eval()
        device = next(self.model.parameters()).device
        y_tensor = torch.from_numpy(y).long().to(device)

        y_pred_probs = []

        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch_x = X[i:i + batch_size]
                batch_x_tensor = torch.tensor(batch_x, dtype=torch.float32).to(device)

                if struct_data is not None:
                    batch_struct = struct_data[i:i + batch_size]
                    batch_struct = Batch.from_data_list(batch_struct).to(device)
                    output = self.model(batch_x_tensor, batch_struct)
                else:
                    output = self.model(batch_x_tensor)

                if isinstance(output, tuple):
                    output = output[0]

                output = F.softmax(output, dim=1)
                y_pred_probs.extend(output[:, 1].detach().cpu().numpy())

        pred_probs_tensor = torch.tensor(y_pred_probs, dtype=torch.float32).to(device)
        loss = self.loss_f(torch.stack([1 - pred_probs_tensor, pred_probs_tensor], dim=1), y_tensor).item()
        auc = roc_auc_score(y, np.array(y_pred_probs))

        return loss, auc

    def _accuracy(self, y_pred, y):
        return float(sum(y_pred == y)) / y.shape[0]

    def predict(self, X):
        X = Variable(torch.from_numpy(X.astype(np.float32)))
        if cuda:
            X= X.cuda()        
        y_pred = self.model(X)
        return y_pred        

    def predict_proba(self, X):
        self.model.eval()
        return self.model.predict_proba(X)

class RNA_Dataset(Dataset):
    def __init__(self, seq_data, labels, struct_data):
        self.seq_data = seq_data
        self.labels = labels
        self.struct_data = struct_data

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.seq_data[idx], self.labels[idx], self.struct_data[idx]

def custom_collate(batch):
    seq_batch = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch])
    struct_batch = Batch.from_data_list([item[2] for item in batch])
    return (seq_batch, labels), struct_batch

class GNNDiffPool(nn.Module):
    def __init__(self, in_channels=8, hidden_channels=64, assign_ratio=0.25):
        super().__init__()
        self.embed_gnn1 = GCNConv(in_channels, hidden_channels)
        self.embed_gnn2 = GCNConv(hidden_channels, hidden_channels)

        self.assign_gnn1 = GCNConv(in_channels, hidden_channels)
        self.assign_gnn2 = GCNConv(hidden_channels, int(assign_ratio * 100))

        self.output_dim = hidden_channels

    def forward(self, data):
        x = data.x.float()
        edge_index = data.edge_index
        batch = data.batch

        z = F.relu(self.embed_gnn1(x, edge_index))  #
        z = F.relu(self.embed_gnn2(z, edge_index))  #

        s = F.relu(self.assign_gnn1(x, edge_index))  #
        s = F.softmax(self.assign_gnn2(s, edge_index), dim=-1)  #


        z_dense, mask = to_dense_batch(z, batch)       # [B, N, H]
        s_dense, _ = to_dense_batch(s, batch)          # [B, N, K]
        adj = to_dense_adj(edge_index, batch=batch)    # [B, N, N]

        out, _, _, _ = dense_diff_pool(z_dense, adj, s_dense, mask)  # [B, K, H]
        return out.mean(dim=1)  # [B, hidden_channels]

class CNN(nn.Module):
    def __init__(self, nb_filter, channel, num_classes=2, kernel_size=(4, 10), pool_size=(1, 3),
                 labcounts=32, window_size=12, hidden_size=200, stride=(1, 1), padding=0):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(channel, nb_filter, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(nb_filter),
            nn.ReLU())
        self.pool1 = nn.MaxPool2d(pool_size, stride=stride)

        out1_size = int((window_size + 2*padding - (kernel_size[1] - 1) - 1)/stride[1] + 1)
        maxpool_size = int((out1_size + 2*padding - (pool_size[1] - 1) - 1)/stride[1] + 1)

        self.layer2 = nn.Sequential(
            nn.Conv2d(nb_filter, nb_filter, kernel_size=(1, 10), stride=stride, padding=padding),
            nn.BatchNorm2d(nb_filter),
            nn.ReLU(),
            nn.MaxPool2d(pool_size, stride=stride))

        out2_size = int((maxpool_size + 2*padding - (10 - 1) - 1)/stride[1] + 1)
        maxpool2_size = int((out2_size + 2*padding - (pool_size[1] - 1) - 1)/stride[1] + 1)

        self.drop1 = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(maxpool2_size * nb_filter, hidden_size)
        self.drop2 = nn.Dropout(p=0.3)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

        self.output_dim = hidden_size  # for feature fusion

    def extract_features(self, x):
        out = self.layer1(x)
        out = self.pool1(out)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.drop1(out)
        out = self.fc1(out)
        return out  # shape [B, hidden_size]

    def forward(self, x):
        out = self.extract_features(x)
        out = self.drop2(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = torch.sigmoid(out)
        return out

    def layer1out(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x.astype(np.float32))
        with torch.no_grad():
            x = Variable(x)
        # x = Variable(x, volatile=True)
        if cuda:
            x = x.cuda()
        out = self.layer1(x)
        temp = out.data.cpu().numpy()
        return temp
    
    def predict_proba(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x.astype(np.float32))
        with torch.no_grad():
            x = Variable(x)
        # x = Variable(x, volatile=True)
        if cuda:
            x = x.cuda()
        y = self.forward(x)
        temp = y.data.cpu().numpy()
        return temp[:, 1]

class CNN_Tran(nn.Module):
    def __init__(self, nb_filter=16, channel=7, num_classes=2,
                 kernel_size=(4, 10), pool_size=(1, 3), labcounts=32,
                 window_size=101, hidden_size=200, stride=(1, 1),
                 padding=0, nhead=4):
        super(CNN_Tran, self).__init__()

        #
        self.layer1 = nn.Sequential(
            nn.Conv2d(channel, nb_filter, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(nb_filter),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(pool_size, stride=stride)

        #
        self.layer2 = nn.Sequential(
            nn.Conv2d(nb_filter, nb_filter, kernel_size=(1, 10), stride=stride, padding=padding),
            nn.BatchNorm2d(nb_filter),
            nn.ReLU(),
            nn.MaxPool2d(pool_size, stride=stride)
        )

        #
        out1_size = int((window_size + 2*padding - (kernel_size[1] - 1) - 1)/stride[1] + 1)
        maxpool1 = int((out1_size + 2*padding - (pool_size[1] - 1) - 1)/stride[1] + 1)
        out2_size = int((maxpool1 + 2*padding - (10 - 1) - 1)/stride[1] + 1)
        maxpool2 = int((out2_size + 2*padding - (pool_size[1] - 1) - 1)/stride[1] + 1)

        self.drop1 = nn.Dropout(p=0.3)
        self.cnn_fc = nn.Linear(maxpool2 * nb_filter, hidden_size)

        # Transformer
        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size,
            dropout=0.1,
            activation='gelu'
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=1)

        self.drop2 = nn.Dropout(p=0.3)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

        self.output_dim = hidden_size  # 用于特征拼接

    def extract_features(self, x):
        x = self.layer1(x)
        x = self.pool1(x)
        x = self.layer2(x)                # [B, nb_filter, H=1, W]
        x = x.view(x.size(0), 1, -1)      # [B, 1, nb_filter * W]
        x = self.drop1(x)
        x = self.cnn_fc(x.squeeze(1))     # [B, hidden_size]
        x = x.unsqueeze(0)                # [1, B, hidden_size]
        x = self.transformer(x)           # [1, B, hidden_size]
        return x.squeeze(0)               # [B, hidden_size]

    def forward(self, x):
        f = self.extract_features(x)
        f = self.drop2(f)
        f = self.relu1(f)
        f = self.fc2(f)
        return torch.sigmoid(f)

    def layer1out(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x.astype(np.float32))
        with torch.no_grad():
            x = Variable(x)
        # x = Variable(x, volatile=True)
        if cuda:
            x = x.cuda()
        out = self.layer1(x)
        temp = out.data.cpu().numpy()
        return temp
        
    def predict_proba(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x.astype(np.float32))
        with torch.no_grad():
            x = Variable(x)
        #x = Variable(x, volatile=True)
        if cuda:
            x = x.cuda()
        y = self.forward(x)
        temp = y.data.cpu().numpy()
        return temp[:, 1]

def convR(in_channels, out_channels, kernel_size, stride=1, padding = (0, 1)):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                     padding=padding, stride=stride, bias=False)

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channel, nb_filter = 16, kernel_size = (1, 3), stride=1, downsample=None): #滤波器16 卷积核高度1宽度3
        super(ResidualBlock, self).__init__()
        self.conv1 = convR(in_channel, nb_filter, kernel_size = kernel_size, stride = stride)
        self.bn1 = nn.BatchNorm2d(nb_filter)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = convR(nb_filter, nb_filter, kernel_size = kernel_size, stride = stride)
        self.bn2 = nn.BatchNorm2d(nb_filter)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# ResNet Module
class ResNet(nn.Module):
    def __init__(self, block, layers, nb_filter=16, channel=7, labcounts=12,
                 window_size=36, kernel_size=(1, 3), pool_size=(1, 3),
                 num_classes=2, hidden_size=200):
        super(ResNet, self).__init__()
        self.in_channels = channel
        self.conv = convR(self.in_channels, nb_filter, kernel_size=(4, 10))
        cnn1_size = window_size - 7
        self.bn = nn.BatchNorm2d(nb_filter)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self.make_layer(block, nb_filter, layers[0], kernel_size=kernel_size)
        self.layer2 = self.make_layer(block, nb_filter * 2, layers[1], 1, kernel_size=kernel_size, in_channels=nb_filter)
        self.layer3 = self.make_layer(block, nb_filter * 4, layers[2], 1, kernel_size=kernel_size, in_channels=2 * nb_filter)

        self.avg_pool = nn.AvgPool2d(pool_size)
        avgpool2_1_size = int((cnn1_size - (pool_size[1] - 1) - 1) / pool_size[1]) + 1
        last_layer_size = 4 * nb_filter * avgpool2_1_size
        self.fc = nn.Linear(last_layer_size, hidden_size)
        self.drop2 = nn.Dropout(p=0.3)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

        self.output_dim = hidden_size  #

    def make_layer(self, block, out_channels, blocks, stride=1, kernel_size=(1, 10), in_channels=16):
        downsample = None
        if (stride != 1) or (in_channels != out_channels):
            downsample = nn.Sequential(
                convR(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = [block(in_channels, out_channels, kernel_size=kernel_size, stride=stride, downsample=downsample)]
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels, kernel_size=kernel_size))
        return nn.Sequential(*layers)

    def extract_features(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out  # shape = [B, hidden_size]

    def forward(self, x):
        out = self.extract_features(x)
        out = self.drop2(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return torch.sigmoid(out)

    def layer1out(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x.astype(np.float32))
        with torch.no_grad():
            x = Variable(x)
        #x = Variable(x, volatile=True)
        if cuda:
            x = x.cuda()
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        temp = out.data.cpu().numpy()
        return temp

    def predict_proba(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x.astype(np.float32))
        with torch.no_grad():
            x = Variable(x)
        #x = Variable(x, volatile=True)
        if cuda:
            x = x.cuda()
        y = self.forward(x)
        temp = y.data.cpu().numpy()
        return temp[:, 1]

class MultiBranchModel(nn.Module):
    def __init__(self, cnn, cnn_tran, resnet,gnn, use_gnn=True, gnn_output_dim=64):
        super().__init__()
        self.cnn = cnn
        self.cnn_tran = cnn_tran
        self.resnet = resnet
        self.gnn = gnn
        self.use_gnn = use_gnn

        fusion_dim = cnn.output_dim + gnn_output_dim

        self.fc1 = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )
        self.att_fusion = nn.Sequential(
            nn.Linear(2 * 3, 64),  # f1, f2, f3 是 [B, 2]，拼接后为 [B, 6]
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Softmax(dim=1)
        )

    def forward(self, seq_x, struct_data=None):
        if self.use_gnn and struct_data is not None:
            struct_data = struct_data.to(seq_x.device)
            struct_feat = self.gnn(struct_data)
        else:
            struct_feat = torch.zeros(seq_x.size(0), 64).to(seq_x.device)
        cnn_feat = self.cnn.extract_features(seq_x)
        tran_feat = self.cnn_tran.extract_features(seq_x)
        res_feat = self.resnet.extract_features(seq_x)

        f1 = torch.cat([cnn_feat, struct_feat], dim=1)
        f2 = torch.cat([tran_feat, struct_feat], dim=1)
        f3 = torch.cat([res_feat, struct_feat], dim=1)

        f1_out = self.fc1(f1)  # [B, 2]
        f2_out = self.fc2(f2)
        f3_out = self.fc3(f3)

        #
        out = (f1_out + f2_out +  f3_out) / 3   # [B, 2]
        if self.training:
            return out  #
        else:
            return out, f1_out[:, 1], f2_out[:, 1], f3_out[:, 1]

    def predict_proba(self, seq_x, struct_data=None):
        self.eval()
        all_probs = []
        with torch.no_grad():
            for i in range(0, len(seq_x), 100):
                batch_x = seq_x[i:i + 100]
                batch_struct = struct_data[i:i + 100] if struct_data else None
                batch_x_tensor = torch.tensor(batch_x, dtype=torch.float32).to(next(self.parameters()).device)

                probs = self(batch_x_tensor, batch_struct)
                probs = probs[:, 1].detach().cpu().numpy()
                all_probs.extend(probs)
        return np.array(all_probs)

def train_network(model_type, X_train, y_train, channel=7, window_size=107, model_file='model.pkl', batch_size=100, n_epochs=50, num_filters=16, motif=False, motif_seqs=[], motif_outdir='motifs', learning_rate=0.0001, weight_decay=0.00001, struct_data_list=None,validation_data=None,
                  struct_val_data=None):
    print('model training for ', model_type)
    if model_type == 'MultiBranch':
        cnn = CNN(nb_filter=num_filters, labcounts=4, window_size=window_size, channel=channel)
        cnn_tran = CNN_Tran(nb_filter=num_filters, window_size=window_size, channel=channel, hidden_size=200, nhead=4)
        resnet = ResNet(ResidualBlock, [3, 3, 3], nb_filter=num_filters, labcounts=4, channel=channel, window_size=window_size)
        gnn = GNNDiffPool(in_channels=6, hidden_channels=64)
        model = MultiBranchModel(cnn=cnn, cnn_tran=cnn_tran, resnet=resnet, gnn=gnn, use_gnn=(struct_data_list is not None))
    else:
        print('only support CNN, CNN-Tran, ResNet, MultiBranch model')
        return

    if cuda:
        model = model.cuda()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=weight_decay
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

    clf = Estimator(model, model_file=model_file)
    clf.compile(optimizer=optimizer, loss=nn.CrossEntropyLoss())

    loss = clf.fit(
        X_train, y_train,
        batch_size=batch_size,
        nb_epoch=n_epochs,
        validation_data=validation_data,
        struct_data=struct_data_list,
        struct_val_data = struct_val_data
    )

    scheduler.step()

    torch.save(model.state_dict(), model_file)
    return loss


def predict_network(model_type, X_test, channel, window_size, model_file='model.pkl', batch_size=100, n_epochs=50, num_filters=16, struct_data=None,return_branches=False):
    print('model predict for ', model_type)

    if model_type == 'MultiBranch':
        cnn = CNN(nb_filter=num_filters, labcounts=4, window_size=window_size, channel=channel)
        cnn_tran = CNN_Tran(nb_filter=num_filters, window_size=window_size, channel=channel, hidden_size=200, nhead=4)
        resnet = ResNet(ResidualBlock, [3, 3, 3], nb_filter=num_filters, labcounts=4, channel=channel, window_size=window_size)
        gnn = GNNDiffPool(in_channels=6, hidden_channels=64)
        model = MultiBranchModel(cnn, cnn_tran, resnet, gnn, use_gnn=(struct_data is not None))
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    if model is None:
        raise RuntimeError("Model initialization failed")

    if cuda:
        model = model.cuda()

    best_path = f"{model_file}.best_auc.pth"
    if os.path.exists(best_path):
        print(f"Loading the validation set performs the optimal model:{best_path}")
        model.load_state_dict(torch.load(best_path))
    else:
        print(f"{best_path} not found, load default model：{model_file}")
        model.load_state_dict(torch.load(model_file))
    model.eval()

    try:
        if model_type == "MultiBranch":
            X_tensor = torch.tensor(X_test, dtype=torch.float32).to(next(model.parameters()).device)
            struct_tensor = Batch.from_data_list(struct_data).to(X_tensor.device)

            with torch.no_grad():
                out, p1, p2, p3 = model(X_tensor, struct_tensor)
                out = F.softmax(out, dim=1)
            probs = out[:, 1].detach().cpu().numpy()
            p1 = p1.detach().cpu().numpy()
            p2 = p2.detach().cpu().numpy()
            p3 = p3.detach().cpu().numpy()
        else:
            probs = model.predict_proba(X_test)
    except Exception as e:
        print(f"[EXCEPT] fallback to manual batching: {e}")
        # === fallback 分批手动推理 ===
        pred = []
        for i in range(0, len(X_test), batch_size):
            batch_x = X_test[i:i + batch_size]
            batch_x_tensor = torch.tensor(batch_x, dtype=torch.float32).to(next(model.parameters()).device)
            if model_type == "MultiBranch":
                batch_struct_list = struct_data[i:i + batch_size]
                batch_struct = Batch.from_data_list(batch_struct_list).to(next(model.parameters()).device)
                batch_pred = model(batch_x_tensor, batch_struct)
                batch_pred = F.softmax(batch_pred, dim=1)
            else:
                batch_pred = model(batch_x_tensor)

            batch_pred = batch_pred[:, 1].detach().cpu().numpy()
            pred.extend(batch_pred)
        probs = np.array(pred)

    if return_branches:
        return probs, p1, p2, p3
    else:
        return probs


def batch(tensor, batch_size = 1000):
    tensor_list = []
    length = tensor.shape[0]
    i = 0
    while True:
        if (i+1) * batch_size >= length:
            tensor_list.append(tensor[i * batch_size: length])
            return tensor_list
        tensor_list.append(tensor[i * batch_size: (i+1) * batch_size])
        i += 1


def run(parser):
    posi = parser.posi #
    nega = parser.nega #
    out_file = parser.out_file #
    train = parser.train #
    model_file = parser.model_file #
    predict = parser.predict  #
    batch_size = parser.batch_size #
    n_epochs = parser.n_epochs #
    num_filters = parser.num_filters
    testfile = parser.testfile #
    motif = parser.motif #
    motif_outdir = parser.motif_dir #
    if args.posi and args.posi != "None":
        dataset_path = args.posi
    elif args.testfile and args.testfile != "None":
        dataset_path = args.testfile
    else:
        raise ValueError("Unable to extract dataset name: make sure --posi or --testfile provides at least one")
    dataset_name = os.path.basename(dataset_path).split('.')[0]

    if predict:
        train = False
        if testfile == '':
            print ('you need specify the fasta file for predicting when predict is True')
            return
    if train:
        if posi == '' or nega == '':
            print ('you need specify the training positive and negative fasta file for training when train is True')
            return

    if train:
        data = read_data_file(posi, nega)
        motif_seqs = data['seq']

        configs = [
            (7, 101), (4, 151), (3, 201),
            (2, 251), (2, 301), (2, 351),
            (1, 401), (1, 451), (1, 501),
        ]
        model_types = ["MultiBranch"]

        for ch, ws in configs:
            print(f"{ws}")

            train_bags, train_labels = get_data(posi, nega, channel=ch, window_size=ws)
            from sklearn.model_selection import train_test_split

            X = np.array(train_bags)
            Y = np.array(train_labels)
            struct_data_list = load_struct_graphs(motif_seqs, dataset_name=dataset_name, use_rnafold=True)


            X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

            if struct_data_list is not None:
                struct_train, struct_val = train_test_split(struct_data_list, test_size=0.2, random_state=42)
            else:
                struct_train, struct_val = None, None
            for mt in model_types:
                mt_name = mt.replace("-", "")
                model_file = f"{mt_name}.{ws}"

            train_network(
                mt, X_train, y_train,
                channel=ch,
                window_size=ws + 6,
                model_file=model_file,
                batch_size=batch_size,
                n_epochs=n_epochs,
                num_filters=num_filters,
                motif=False,
                motif_seqs=motif_seqs,
                motif_outdir=motif_outdir,
                struct_data_list=struct_train,
                validation_data=(X_val, y_val),
                struct_val_data=struct_val
            )
        # ------------------------------------------------------------
        print("Done！")

    elif predict:

        mb_windows = [(7, 101), (4, 151), (3, 201), (2, 251), (2, 301), (2, 351), (1, 401), (1, 451), (1, 501)]
        mb_preds_att = []
        mb_preds_dbo = []
        mb_preds_auc = []

        for ch, ws in mb_windows:
            print(f"MultiBranch predicting on window size {ws}")

            X_test, X_labels = get_data(testfile, nega, channel=ch, window_size=ws)
            X_test = np.array(X_test)
            seqs = read_seq_graphprot(testfile)[0] + read_seq_graphprot(nega)[0]
            struct_data = load_struct_graphs(seqs, dataset_name=dataset_name + "_eval", use_rnafold=True)

            pred_att, p1, p2, p3 = predict_network(
                model_type="MultiBranch",
                X_test=X_test,
                channel=ch,
                window_size=ws + 6,
                model_file=f"MultiBranch.{ws}",
                batch_size=batch_size,
                n_epochs=n_epochs,
                num_filters=num_filters,
                struct_data=struct_data,
                return_branches=True
            )

            from DBO import DBO
            pred_dbo, w1, w2, w3 = DBO(p1, p2, p3, X_labels)
            pred = w1 * p1 + w2 * p2 + w3 * p3

            mb_preds_att.append(pred_att)
            mb_preds_auc.append(pred_dbo)
            mb_preds_dbo.append(pred)
        MultiBranchAvg = sum(mb_preds_att) / len(mb_preds_att)
        auc2 = max(mb_preds_auc)
        MultiBranchAvg1 = sum(mb_preds_dbo) / len(mb_preds_dbo)
        with open("MultiBranch_9window_eval.txt", 'a') as fw:
            fw.write(f"\nA new one-off experiment: the MultiBranch Multi-Window Fusion Evaluation - Sample:{dataset_name}\n")
            auc1 = roc_auc_score(X_labels, MultiBranchAvg)
            print(f'MultiBranch AUC [{dataset_name}]: {auc1:.3f}')
            print(f'MultiBranch-DBO AUC [{dataset_name}]: {auc2:.3f}')
            fw.write(f'MultiBranch AUC ({dataset_name}): {auc1:.4f}\n')
            fw.write(f'MultiBranch-DBO AUC ({dataset_name}): {auc2:.4f}\n')

        for name, pred in [("MultiBranch", MultiBranchAvg)]:
            evaluate_all_metrics(
                X_labels,
                pred,
                output_path="MultiBranch_9window_eval.txt",
                title=f"MultiBranch-9window:{dataset_name} ({name})"
            )
        for name, pred in [("MultiBranch-DBO", MultiBranchAvg1)]:
            evaluate_all_metrics(
                X_labels,
                pred,
                output_path="MultiBranch_9window_eval.txt",
                title=f"MultiBranch-9window-DBO:{dataset_name} ({name})"
            )
    else:
        print ('please specify that you want to train the mdoel or predict for your own sequences')


def parse_arguments(parser):
    parser.add_argument('--posi', type=str, metavar='<postive_sequecne_file>', help='The fasta file of positive training samples')
    parser.add_argument('--nega', type=str, metavar='<negative_sequecne_file>', help='The fasta file of negative training samples')
    parser.add_argument('--out_file', type=str, default='prediction.txt', help='The output file used to store the prediction probability of the testing sequences')
    parser.add_argument('--train', type=bool, default=False, help='The path to the Pickled file containing durations between visits of patients. If you are not using duration information, do not use this option')
    parser.add_argument('--model_file', type=str, default='model.pkl', help='The file to save model parameters. Use this option if you want to train on your sequences or predict for your sequences')
    parser.add_argument('--predict', type=bool, default=False,  help='Predicting the RNA-protein binding sites for your input sequences, if using train, then it will be False')
    parser.add_argument('--testfile', type=str, default='',  help='the test fast file for sequences you want to predict for, you need specify it when using predict')
    parser.add_argument('--batch_size', type=int, default=256, help='The size of a single mini-batch (default value: 256)')
    parser.add_argument('--num_filters', type=int, default=16, help='The number of filters for CNNs (default value: 16)')
    parser.add_argument('--n_epochs', type=int, default=30, help='The number of training epochs (default value: 30)')
    parser.add_argument('--motif', type=bool, default=False, help='It is used to identify binding motifs from sequences.')
    parser.add_argument('--motif_dir', type=str, default='motifs', help='The dir used to store the prediction binding motifs.')
    args = parser.parse_args()
    return args

parser = argparse.ArgumentParser()
args = parse_arguments(parser)
print(args)
run(args)
