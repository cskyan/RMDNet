#-- coding:utf8 --
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
import numpy as np
np.set_printoptions(threshold=np.inf)
import pdb
sys.path.append('/home/.../PhD/DBODL')
import torch    
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
import argparse
import DBO
from scipy.special import gamma
import os
import math
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'

# GPU or CPU
if torch.cuda.is_available():
        cuda = True
        # torch.cuda.set_device(1)
        print('===> Using GPU')
        torch.cuda.set_device(0)
else:
        cuda = False
        print('===> Using CPU')

# Filling subsequences with base nucleotide N
def padding_sequence_new(seq, window_size = 101, repkey = 'N'):
    seq_len = len(seq)
    new_seq = seq
    if seq_len < window_size:
        gap_len = window_size -seq_len
        new_seq = seq + repkey * gap_len
    return new_seq

# Filling entire sequence with base nucleotide N
def padding_sequence(seq, max_len = 501, repkey = 'N'):
    seq_len = len(seq)
    if seq_len < max_len:
        gap_len = max_len -seq_len
        new_seq = seq + repkey * gap_len
    else:
        new_seq = seq[:max_len]
    return new_seq

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

def split_overlap_seq(seq, window_size):
    overlap_size = 50
    bag_seqs = []
    seq_len = len(seq)
    if seq_len >= window_size:
        num_ins = (seq_len - window_size)/(window_size - overlap_size) + 1
        remain_ins = (seq_len - window_size)%(window_size - overlap_size)
    else:
        num_ins = 0
    bag = []
    end = 0
    for ind in range(int(num_ins)):
        start = end - overlap_size
        if start < 0:
            start = 0
        end = start + window_size
        subseq = seq[start:end]
        bag_seqs.append(subseq)
    if num_ins == 0:
        seq1 = seq
        pad_seq = padding_sequence_new(seq1, window_size)
        bag_seqs.append(pad_seq)
    else:
        if remain_ins > 10:
            seq1 = seq[-window_size:]
            pad_seq = padding_sequence_new(seq1, window_size)
            bag_seqs.append(pad_seq)
    return bag_seqs

# Read sequence file
def read_seq_graphprot(seq_file, label = 1):
    seq_list = []
    labels = []
    seq = ''
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

# Obtain the processed one-hot coding matrix and corresponding labels

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

def get_data(posi, nega = None, channel = 7,  window_size = 101, train = True): #
    data = read_data_file(posi, nega, train = train)
    if channel == 1:
        train_bags, label = get_bag_data_1_channel(data, max_len = window_size)

    else:
        train_bags, label = get_bag_data(data, channel = channel, window_size = window_size)
    
    return train_bags, label

class Estimator(object):

    def __init__(self, model):
        self.model = model

    def compile(self, optimizer, loss):
        self.optimizer = optimizer
        self.loss_f = loss

    def _fit(self, train_loader):
        """
        train one epoch
        """
        loss_list = []
        for idx, (X, y) in enumerate(train_loader):
            X_v = Variable(X)
            y_v = Variable(y)
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

    def fit(self, X, y, batch_size=32, nb_epoch=10, validation_data=()):
        print (X.shape)
        train_set = TensorDataset(torch.from_numpy(X.astype(np.float32)), torch.from_numpy(y.astype(np.float32)).long().view(-1))
        train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
        self.model.train()
        train_loss_print = []
        
        for t in range(nb_epoch):
            loss = self._fit(train_loader)
            train_loss_print.append(loss)
            print("Epoch %s/%s loss: %06.4f" % (t, nb_epoch, loss))
        return loss
       


    def evaluate(self, X, y, batch_size):
        
        y_pred = self.predict(X)

        y_v = Variable(torch.from_numpy(y).long(), requires_grad=False)
        if cuda:
            y_v = y_v.cuda()
        loss = self.loss_f(y_pred, y_v)
        predict = y_pred.data.cpu().numpy()[:, 1].flatten()
        auc = roc_auc_score(y, predict)

        return loss.item(), auc

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


class CNN(nn.Module):
    def __init__(self,  nb_filter, channel , num_classes = 2, kernel_size = (4, 10), pool_size = (1, 3), labcounts = 32, window_size = 12, hidden_size = 200, stride = (1, 1), padding = 0):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(channel, nb_filter, kernel_size, stride = stride, padding = padding),
            nn.BatchNorm2d(nb_filter),
            nn.ReLU())
        self.pool1 = nn.MaxPool2d(pool_size, stride = stride)
        out1_size = int((window_size + 2*padding - (kernel_size[1] - 1) - 1)/stride[1] + 1)
        maxpool_size = int((out1_size + 2*padding - (pool_size[1] - 1) - 1)/stride[1] + 1)
        self.layer2 = nn.Sequential(
            nn.Conv2d(nb_filter, nb_filter, kernel_size = (1, 10), stride = stride, padding = padding),
            nn.BatchNorm2d(nb_filter),
            nn.ReLU(),
            nn.MaxPool2d(pool_size, stride = stride))
        out2_size = int((maxpool_size + 2*padding - (kernel_size[1] - 1) - 1)/stride[1] + 1)
        maxpool2_size = int((out2_size + 2*padding - (pool_size[1] - 1) - 1)/stride[1] + 1)
        self.drop1 = nn.Dropout(p=0.25)
        # print ('maxpool_size', maxpool_size)
        self.fc1 = nn.Linear(maxpool2_size*nb_filter, hidden_size)
        self.drop2 = nn.Dropout(p=0.25)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.pool1(out)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.drop1(out)
        out = self.fc1(out)
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
        if cuda:
            x = x.cuda()
        y = self.forward(x)
        temp = y.data.cpu().numpy()
        return temp[:, 1]

class CNN_LSTM(nn.Module):
    def __init__(self, nb_filter, channel = 7, num_classes = 2, kernel_size = (4, 10), pool_size = (1, 3), labcounts = 12, window_size = 12, hidden_size = 200, stride = (1, 1), padding = 0, num_layers = 2):
        super(CNN_LSTM, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(channel, nb_filter, kernel_size, stride = stride, padding = padding),
            nn.BatchNorm2d(nb_filter),
            nn.ReLU())
        self.pool1 = nn.MaxPool2d(pool_size, stride = stride)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        out1_size = (window_size + 2*padding - (kernel_size[1] - 1) - 1)/stride[1] + 1
        maxpool_size = (out1_size + 2*padding - (pool_size[1] - 1) - 1)/stride[1] + 1
        self.downsample = nn.Conv2d(nb_filter, 1, kernel_size = (1, 10), stride = stride, padding = padding)
        input_size = int((maxpool_size + 2*padding - (kernel_size[1] - 1) - 1)/stride[1]) + 1
        self.layer2 = nn.LSTM(input_size, hidden_size, num_layers, batch_first = True, bidirectional=True) #定义LSTM层
        self.drop1 = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear(2*hidden_size, hidden_size)
        self.drop2 = nn.Dropout(p=0.25)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

        
    def forward(self, x):
        out = self.layer1(x)
        out = self.pool1(out)
        out = self.downsample(out)
        out = torch.squeeze(out, 1)

        if cuda:
            x = x.cuda()
            h0 = Variable(torch.zeros(self.num_layers*2, out.size(0), self.hidden_size)).cuda() 
            # print(h0.shape)
            c0 = Variable(torch.zeros(self.num_layers*2, out.size(0), self.hidden_size)).cuda()
        else:
            h0 = Variable(torch.zeros(self.num_layers*2, out.size(0), self.hidden_size)) 
            c0 = Variable(torch.zeros(self.num_layers*2, out.size(0), self.hidden_size))
        out, _  = self.layer2(out, (h0, c0))
        out = out[:, -1, :]
        out = self.drop1(out)
        out = self.fc1(out)
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
    def __init__(self, in_channel, nb_filter = 16, kernel_size = (1, 3), stride=1, downsample=None):
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
    def __init__(self, block, layers, nb_filter = 16, channel = 7, labcounts = 12, window_size = 36, kernel_size = (1, 3), pool_size = (1, 3), num_classes=2, hidden_size = 200):
        super(ResNet, self).__init__()
        self.in_channels = channel
        self.conv = convR(self.in_channels, nb_filter, kernel_size = (4, 10))
        cnn1_size = window_size - 7
        self.bn = nn.BatchNorm2d(nb_filter)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, nb_filter, layers[0],  kernel_size = kernel_size)
        self.layer2 = self.make_layer(block, nb_filter*2, layers[1], 1, kernel_size = kernel_size, in_channels = nb_filter)
        self.layer3 = self.make_layer(block, nb_filter*4, layers[2], 1, kernel_size = kernel_size, in_channels = 2*nb_filter)
        self.avg_pool = nn.AvgPool2d(pool_size)
        avgpool2_1_size = int((cnn1_size - (pool_size[1] - 1) - 1)/pool_size[1]) + 1
        last_layer_size = 4*nb_filter*avgpool2_1_size
        self.fc = nn.Linear(last_layer_size, hidden_size)
        self.drop2 = nn.Dropout(p=0.25)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def make_layer(self, block, out_channels, blocks, stride=1,  kernel_size = (1, 10), in_channels = 16):
        downsample = None
        if (stride != 1) or (in_channels != out_channels):
            downsample = nn.Sequential(
                convR(in_channels, out_channels, kernel_size = kernel_size, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(in_channels, out_channels, kernel_size = kernel_size, stride = stride, downsample = downsample))
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels, kernel_size = kernel_size))
        return nn.Sequential(*layers)
     
    def forward(self, x):

        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
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
        if cuda:
            x = x.cuda()
        y = self.forward(x)
        temp = y.data.cpu().numpy()
        return temp[:, 1]


def train_network(model_type, X_train, y_train, channel = 7, window_size = 107, model_file = 'model.pkl', batch_size = 100, n_epochs = 50, num_filters = 16, motif = False, motif_seqs = [], motif_outdir = 'motifs', learning_rate=0.0001, weight_decay=0.00001):
    print ('model training for ', model_type)

    if model_type == 'CNN':
        model = CNN(nb_filter =num_filters, labcounts = 4, window_size = window_size, channel = channel)
    elif model_type == 'CNN-LSTM':
        model = CNN_LSTM(nb_filter = num_filters, labcounts = 4, window_size = window_size, channel = channel)
    elif model_type == 'ResNet':
        model = ResNet(ResidualBlock, [3, 3, 3], nb_filter = num_filters, labcounts = 4, channel = channel , window_size = window_size)
    else:
        print ('only support CNN, CNN-python LSTM, ResNet model')

    if cuda:
        model = model.cuda()
    clf = Estimator(model)
    clf.compile(optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay),
                loss=nn.CrossEntropyLoss())
    loss = clf.fit(X_train, y_train, batch_size=batch_size, nb_epoch=n_epochs)
    if motif and channel == 1:
        detect_motifs(model, motif_seqs, X_train, motif_outdir)

    torch.save(model.state_dict(), model_file)
    return loss

def predict_network(model_type, X_test, channel , window_size , model_file = 'model.pkl', batch_size = 100, n_epochs = 50, num_filters = 16):
    print ('model predict for ', model_type)

    if model_type == 'CNN':
        model = CNN(nb_filter =num_filters, labcounts = 4, window_size = window_size, channel = channel)
    elif model_type == 'CNN-LSTM':
        model = CNN_LSTM(nb_filter = num_filters, labcounts = 4, window_size = window_size, channel = channel)
    elif model_type == 'ResNet':
        model = ResNet(ResidualBlock, [3, 3, 3], nb_filter = num_filters, labcounts = 4, channel = channel , window_size = window_size)
    else:
        print ('only support CNN, CNN-LSTM, ResNet model')

    if cuda:
        model = model.cuda()
                
    model.load_state_dict(torch.load(model_file))
    try:
        pred = model.predict_proba(X_test)
    except: #to handle the out-of-memory when testing
        test_batch = batch(X_test)
        pred = []
        for test in test_batch:
            pred_test1 = model.predict_proba(test)[:, 1]
            pred = np.concatenate((pred, pred_test1), axis = 0)
    return pred

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

def detect_motifs(model, test_seqs, X_train, output_dir = 'motifs', channel = 1):
    if channel == 1:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for param in model.parameters():
            layer1_para =  param.data.cpu().numpy()
            break

        N = len(test_seqs)
        if N > 15000: # do need all sequence to generate motifs and avoid out-of-memory
        	sele = 15000
        else:
        	sele = N
        ix_all = np.arange(N)
        np.random.shuffle(ix_all)
        ix_test = ix_all[0:sele]
        
        X_train = X_train[ix_test, :, :, :]
        test_seq = []
        for ind in ix_test:
        	test_seq.append(test_seqs[ind])
        test_seqs = test_seq
        filter_outs = model.layer1out(X_train)[:,:, 0, ]

def Bounds(s, Lb, Ub):
    temp = s
    for i in range(len(s)):
        if temp[i] < Lb[0, i]:
            temp[i] = Lb[0, i]
        elif temp[i] > Ub[0, i]:
            temp[i] = Ub[0, i]

    return temp

def swapfun(ss):
    temp = ss
    o = np.zeros((1,len(temp)))
    for i in range(len(ss)):
        o[0,i]=temp[i]
    return o

def levy(d):
    beta = 1.5

    sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2)) / (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)) ** (
                1 / beta)

    u = np.random.rand(d) * sigma
    v = np.random.rand(d)
    step = np.real(u / (np.abs(v) ** (1 / beta)))

    L = 0.01 * step
    return L

def run(parser):
    posi = parser.posi
    nega = parser.nega
    train = parser.train
    model_file = parser.model_file
    predict = parser.predict
    batch_size = parser.batch_size
    n_epochs = parser.n_epochs
    num_filters = parser.num_filters
    testfile = parser.testfile
    motif = parser.motif
    motif_outdir = parser.motif_dir

    

    if predict:
        train = False
        if testfile == '':
            print ('you need specify the fasta file for predicting when predict is True') #当predict为True时，您需要指定用于预测的FASTA文件。
            return
    if train:
        if posi == '' or nega == '':
            print ('you need specify the training positive and negative fasta file for training when train is True') #当train为True时，您需要指定用于训练的正例和负例FASTA文件。
            return

    if train:
        motif_seqs = []
        data = read_data_file(posi, nega)
        motif_seqs = data['seq']

        pop = 30
        M = 1
        c = [0.0001, 1e-5]
        d = [5e-2, 1e-2]
        dim = 2
        P_percent = 0.2
        pNum = round(pop * P_percent)
        lb = c * np.ones((1, dim))
        ub = d * np.ones((1, dim))
        X = np.zeros((pop, dim))
        X1 = np.zeros((pop, dim))
        X2 = np.zeros((pop, dim))
        for i in range(pop):
            for j in range (dim):
                X[i, j] = X[i, j] + 1e-5 - np.mod(2.2 * np.sin(2 * np.pi * X[i, j]) / (2 * np.pi), 1)
                X1[i, j] = X1[i, j] + 1e-5 - np.mod(2.2 * np.sin(2 * np.pi * X1[i, j]) / (2 * np.pi), 1)
                X2[i, j] = X2[i, j] + 1e-5 - np.mod(2.2 * np.sin(2 * np.pi * X2[i, j]) / (2 * np.pi), 1)
                if j == 1:
                    if X[i,j] > 0.0001:
                        X[i,j] =0.0001
                    if X1[i, j] > 0.0001:
                        X1[i, j] = 0.0001
                    if X2[i,j] > 0.0001:
                        X2[i,j] =0.0001
                    if X[i,j] < 1e-5:
                        X[i, j] = 1e-5
                    if X1[i, j] < 1e-5:
                        X1[i, j] = 1e-5
                    if X2[i, j] < 1e-5:
                        X2[i, j] = 1e-5
                if j ==2:
                    if X[i,j] > 5e-2:
                        X[i,j] = 5e-2
                    if X1[i,j] > 5e-2:
                        X1[i,j] = 5e-2
                    if X2[i, j] > 5e-2:
                        X2[i, j] = 5e-2
                    if X[i,j] < 1e-2:
                        X[i, j] = 1e-2
                    if X1[i,j] < 1e-2:
                        X1[i, j] = 1e-2
                    if X2[i,j] < 1e-2:
                        X2[i, j] = 1e-2

        fit = np.zeros((pop, 1))
        fit1 = np.zeros((pop, 1))
        fit2 = np.zeros((pop, 1))

        for i in range(pop):

            train_bags, train_labels = get_data(posi, nega, channel=7, window_size=101)
            model_type = "CNN"
            loss = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=7,
                                 window_size=101 + 6,
                                 model_file=model_type + '.101', batch_size=batch_size, n_epochs=n_epochs,
                                 num_filters=num_filters, learning_rate=X[i, 0], weight_decay=X[i, 1])
            fit[i, 0] = loss

            train_bags, train_labels = get_data(posi, nega, channel=7, window_size=101)
            model_type = "CNN-LSTM"
            loss1 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=7,
                                  window_size=101 + 6,
                                  model_file=model_type + '.101', batch_size=batch_size, n_epochs=n_epochs,
                                  num_filters=num_filters, learning_rate=X1[i, 0], weight_decay=X1[i, 1])
            fit1[i, 0] = loss1

            train_bags, train_labels = get_data(posi, nega, channel=7, window_size=101)
            model_type = "ResNet"
            loss2 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=7,
                                  window_size=101 + 6,
                                  model_file=model_type + '.101', batch_size=batch_size, n_epochs=n_epochs,
                                  num_filters=num_filters, learning_rate=X2[i, 0], weight_decay=X2[i, 1])
            fit2[i, 0] = loss2
        pFit = fit
        pFit1 = fit1
        pFit2 = fit2
        pX = X
        pX1 = X1
        pX2 = X2
        XX = pX
        XX1 = pX1
        XX2 = pX2
        fMin = np.min(fit[:, 0])
        fMin1 = np.min(fit1[:, 0])
        fMin2 = np.min(fit2[:, 0])
        bestI = np.argmin(fit[:, 0])
        bestI1 = np.argmin(fit1[:, 0])
        bestI2 = np.argmin(fit2[:, 0])
        bestX = X[bestI, :]
        bestX1 = X1[bestI1, :]
        bestX2 = X2[bestI2, :]

        for t in range(M):
            B = np.argmax(pFit[:, 0])
            B1 = np.argmax(pFit1[:, 0])
            B2 = np.argmax(pFit2[:, 0])
            worse = X[B, :]
            worse1 = X1[B1, :]
            worse2 = X2[B2, :]
            r2 = np.random.rand(1)
            for i in range(pNum):
                if r2 < 0.9:
                    a = np.random.rand(1)
                    if a > 0.1:
                        a = 1
                    else:
                        a = -1
                    X[i, :] = pX[i, :] + 0.3 * np.abs(pX[i, :] - worse) + a * 0.1 * (XX[i, :])
                    X1[i, :] = pX1[i, :] + 0.3 * np.abs(pX1[i, :] - worse1) + a * 0.1 * (XX1[i, :])
                    X2[i, :] = pX2[i, :] + 0.3 * np.abs(pX2[i, :] - worse2) + a * 0.1 * (XX2[i, :])
                else:
                    aaa = np.random.randint(180, size=1)
                    if aaa == 0 or aaa == 90 or aaa == 180:
                        X[i, :] = pX[i, :]
                        X1[i, :] = pX1[i, :]
                        X2[i, :] = pX2[i, :]
                    theta = aaa * math.pi / 180
                    X[i, :] = pX[i, :] + math.tan(theta) * np.abs(pX[i, :] - XX[i, :])
                    X1[i, :] = pX1[i, :] + math.tan(theta) * np.abs(pX1[i, :] - XX1[i, :])
                    X2[i, :] = pX2[i, :] + math.tan(theta) * np.abs(pX2[i, :] - XX2[i, :])
                X[i, :] = Bounds(X[i, :], lb, ub)
                X1[i, :] = Bounds(X1[i, :], lb, ub)
                X2[i, :] = Bounds(X2[i, :], lb, ub)
                train_bags, train_labels = get_data(posi, nega, channel=7, window_size=101)
                model_type = "CNN"
                loss = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=7,
                                     window_size=101 + 6,
                                     model_file=model_type + '.101', batch_size=batch_size, n_epochs=n_epochs,
                                     num_filters=num_filters, learning_rate=X[i, 0], weight_decay=X[i, 1])
                fit[i, 0] = loss
                train_bags, train_labels = get_data(posi, nega, channel=7, window_size=101)
                model_type = "CNN-LSTM"
                loss1 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=7,
                                      window_size=101 + 6,
                                      model_file=model_type + '.101', batch_size=batch_size, n_epochs=n_epochs,
                                      num_filters=num_filters, learning_rate=X1[i, 0], weight_decay=X1[i, 1])
                fit1[i, 0] = loss1
                train_bags, train_labels = get_data(posi, nega, channel=7, window_size=101)
                model_type = "ResNet"
                loss2 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=7,
                                      window_size=101 + 6,
                                      model_file=model_type + '.101', batch_size=batch_size, n_epochs=n_epochs,
                                      num_filters=num_filters, learning_rate=X2[i, 0], weight_decay=X2[i, 1])
                fit2[i, 0] = loss2
            bestII = np.argmin(fit[:, 0])
            bestII1 = np.argmin(fit1[:, 0])
            bestII2 = np.argmin(fit2[:, 0])
            bestXX = X[bestII, :]
            bestXX1 = X1[bestII1, :]
            bestXX2 = X2[bestII2, :]
            R = 1 - t / M
            Xnew1 = bestXX * (1 - R)
            Xnew2 = bestXX * (1 + R)
            Xnew1l = bestXX1 * (1 - R)
            Xnew2l = bestXX1 * (1 + R)
            Xnew1r = bestXX2 * (1 - R)
            Xnew2r = bestXX2 * (1 + R)
            Xnew1 = Bounds(Xnew1, lb, ub)
            Xnew2 = Bounds(Xnew2, lb, ub)
            Xnew1l = Bounds(Xnew1l, lb, ub)
            Xnew2l = Bounds(Xnew2l, lb, ub)
            Xnew1r = Bounds(Xnew1r, lb, ub)
            Xnew2r = Bounds(Xnew2r, lb, ub)
            Xnew11 = bestX * (1 - R)
            Xnew22 = bestX * (1 + R)
            Xnew11l = bestX1 * (1 - R)
            Xnew22l = bestX1 * (1 + R)
            Xnew11r = bestX2 * (1 - R)
            Xnew22r = bestX2 * (1 + R)
            Xnew11 = Bounds(Xnew11, lb, ub)
            Xnew22 = Bounds(Xnew22, lb, ub)
            Xnew11l = Bounds(Xnew11l, lb, ub)
            Xnew22l = Bounds(Xnew22l, lb, ub)
            Xnew11r = Bounds(Xnew11r, lb, ub)
            Xnew22r = Bounds(Xnew22r, lb, ub)
            xLB = swapfun(Xnew1)
            xUB = swapfun(Xnew2)
            xLB1 = swapfun(Xnew1l)
            xUB1 = swapfun(Xnew2l)
            xLB2 = swapfun(Xnew1r)
            xUB2 = swapfun(Xnew2r)
            for i in range(pNum + 1, 12):
                X[i, :] = bestXX + (np.random.rand(1, dim)) * (pX[i, :] - Xnew1) + (np.random.rand(1, dim)) * (
                        pX[i, :] - Xnew2)
                X1[i, :] = bestXX1 + (np.random.rand(1, dim)) * (pX1[i, :] - Xnew1l) + (np.random.rand(1, dim)) * (
                        pX1[i, :] - Xnew2l)
                X2[i, :] = bestXX2 + (np.random.rand(1, dim)) * (pX2[i, :] - Xnew1r) + (np.random.rand(1, dim)) * (
                        pX2[i, :] - Xnew2r)
                X[i, :] = Bounds(X[i, :], xLB, xUB)
                X1[i, :] = Bounds(X1[i, :], xLB1, xUB1)
                X2[i, :] = Bounds(X2[i, :], xLB2, xUB2)
                train_bags, train_labels = get_data(posi, nega, channel=7, window_size=101)
                model_type = "CNN"
                loss = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=7,
                                     window_size=101 + 6,
                                     model_file=model_type + '.101', batch_size=batch_size, n_epochs=n_epochs,
                                     num_filters=num_filters, learning_rate=X[i, 0], weight_decay=X[i, 1])
                fit[i, 0] = loss
                train_bags, train_labels = get_data(posi, nega, channel=7, window_size=101)
                model_type = "CNN-LSTM"
                loss1 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=7,
                                      window_size=101 + 6,
                                      model_file=model_type + '.101', batch_size=batch_size, n_epochs=n_epochs,
                                      num_filters=num_filters, learning_rate=X1[i, 0], weight_decay=X1[i, 1])
                fit1[i, 0] = loss1
                train_bags, train_labels = get_data(posi, nega, channel=7, window_size=101)
                model_type = "ResNet"
                loss2 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=7,
                                      window_size=101 + 6,
                                      model_file=model_type + '.101', batch_size=batch_size, n_epochs=n_epochs,
                                      num_filters=num_filters, learning_rate=X2[i, 0], weight_decay=X2[i, 1])
                fit2[i, 0] = loss2
            for i in range(13, 19):
                X[i, :] = pX[i, :] + ((np.random.randn(1)) * (pX[i, :] - Xnew11) + (
                        (np.random.rand(1, dim)) * (pX[i, :] - Xnew22)))
                X1[i, :] = pX1[i, :] + ((np.random.randn(1)) * (pX1[i, :] - Xnew11l) + (
                        (np.random.rand(1, dim)) * (pX1[i, :] - Xnew22l)))
                X2[i, :] = pX2[i, :] + ((np.random.randn(1)) * (pX2[i, :] - Xnew11r) + (
                        (np.random.rand(1, dim)) * (pX2[i, :] - Xnew22r)))
                X[i, :] = Bounds(X[i, :], lb, ub)
                X1[i, :] = Bounds(X1[i, :], lb, ub)
                X2[i, :] = Bounds(X2[i, :], lb, ub)
                train_bags, train_labels = get_data(posi, nega, channel=7, window_size=101)
                model_type = "CNN"
                loss = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=7,
                                     window_size=101 + 6,
                                     model_file=model_type + '.101', batch_size=batch_size, n_epochs=n_epochs,
                                     num_filters=num_filters, learning_rate=X[i, 0], weight_decay=X[i, 1])
                fit[i, 0] = loss
                train_bags, train_labels = get_data(posi, nega, channel=7, window_size=101)
                model_type = "CNN-LSTM"
                loss1 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=7,
                                      window_size=101 + 6,
                                      model_file=model_type + '.101', batch_size=batch_size, n_epochs=n_epochs,
                                      num_filters=num_filters, learning_rate=X1[i, 0], weight_decay=X1[i, 1])
                fit1[i, 0] = loss1
                train_bags, train_labels = get_data(posi, nega, channel=7, window_size=101)
                model_type = "ResNet"
                loss2 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=7,
                                      window_size=101 + 6,
                                      model_file=model_type + '.101', batch_size=batch_size, n_epochs=n_epochs,
                                      num_filters=num_filters, learning_rate=X2[i, 0], weight_decay=X2[i, 1])
                fit2[i, 0] = loss2
            for j in range(20, pop):
                X[j, :] = levy(dim)*bestX + np.random.randn(1, dim) * (np.abs(pX[j, :] - bestXX) + np.abs(pX[j, :] - bestX)) / 2
                X1[j, :] = levy(dim)*bestX1 + np.random.randn(1, dim) * (
                            np.abs(pX1[j, :] - bestXX1) + np.abs(pX1[j, :] - bestX1)) / 2
                X2[j, :] = levy(dim)*bestX2 + np.random.randn(1, dim) * (
                            np.abs(pX2[j, :] - bestXX2) + np.abs(pX2[j, :] - bestX2)) / 2
                X[j, :] = Bounds(X[j, :], lb, ub)
                X1[j, :] = Bounds(X1[j, :], lb, ub)
                X2[j, :] = Bounds(X2[j, :], lb, ub)
                train_bags, train_labels = get_data(posi, nega, channel=7, window_size=101)
                model_type = "CNN"
                loss = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=7,
                                     window_size=101 + 6,
                                     model_file=model_type + '.101', batch_size=batch_size, n_epochs=n_epochs,
                                     num_filters=num_filters, learning_rate=X[j, 0], weight_decay=X[j, 1])
                fit[j, 0] = loss
                train_bags, train_labels = get_data(posi, nega, channel=7, window_size=101)
                model_type = "CNN-LSTM"
                loss1 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=7,
                                      window_size=101 + 6,
                                      model_file=model_type + '.101', batch_size=batch_size, n_epochs=n_epochs,
                                      num_filters=num_filters, learning_rate=X1[j, 0], weight_decay=X1[j, 1])
                fit1[j, 0] = loss1
                train_bags, train_labels = get_data(posi, nega, channel=7, window_size=101)
                model_type = "ResNet"
                loss2 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=7,
                                      window_size=101 + 6,
                                      model_file=model_type + '.101', batch_size=batch_size, n_epochs=n_epochs,
                                      num_filters=num_filters, learning_rate=X2[j, 0], weight_decay=X2[j, 1])
                fit2[j, 0] = loss2
            XX = pX
            XX1 = pX1
            XX2 = pX2
            for i in range(pop):
                if fit[i, 0] < pFit[i, 0]:
                    pFit[i, 0] = fit[i, 0]
                    pX[i, :] = X[i, :]
                if pFit[i, 0] < fMin:
                    fMin = pFit[i, 0]
                    bestX = pX[i, :]
            for i in range(pop):
                if fit1[i, 0] < pFit1[i, 0]:
                    pFit1[i, 0] = fit1[i, 0]
                    pX1[i, :] = X1[i, :]
                if pFit1[i, 0] < fMin1:
                    fMin1 = pFit1[i, 0]
                    bestX1 = pX1[i, :]
            for i in range(pop):
                if fit2[i, 0] < pFit2[i, 0]:
                    pFit2[i, 0] = fit2[i, 0]
                    pX2[i, :] = X2[i, :]
                if pFit2[i, 0] < fMin2:
                    fMin2 = pFit2[i, 0]
                    bestX2 = pX2[i, :]
            X = pX
            X1 = pX1
            X2 = pX2

        print("101")
        train_bags, train_labels = get_data(posi, nega, channel = 7, window_size = 101)
        model_type = "CNN"
        train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 7, window_size = 101 + 6, model_file = model_type + '.101', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters, learning_rate=bestX[0], weight_decay=bestX[1])
        model_type = "CNN-LSTM"
        train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 7, window_size = 101 + 6, model_file = model_type + '.101', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters,learning_rate=bestX1[0], weight_decay=bestX1[1])
        model_type = "ResNet"
        train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 7, window_size = 101 + 6, model_file = model_type + '.101', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters, learning_rate=bestX2[0], weight_decay=bestX2[1])

        pop = 30
        M = 1
        c = [0.0001, 1e-5]
        d = [5e-2, 1e-2]
        dim = 2
        P_percent = 0.2
        pNum = round(pop * P_percent)
        lb = c * np.ones((1, dim))
        ub = d * np.ones((1, dim))
        X = np.zeros((pop, dim))
        X1 = np.zeros((pop, dim))
        X2 = np.zeros((pop, dim))
        for i in range(pop):
            for j in range (dim):
                X[i, j] = X[i, j] + 1e-5 - np.mod(2.2 * np.sin(2 * np.pi * X[i, j]) / (2 * np.pi), 1)
                X1[i, j] = X1[i, j] + 1e-5 - np.mod(2.2 * np.sin(2 * np.pi * X1[i, j]) / (2 * np.pi), 1)
                X2[i, j] = X2[i, j] + 1e-5 - np.mod(2.2 * np.sin(2 * np.pi * X2[i, j]) / (2 * np.pi), 1)
                if j == 1:
                    if X[i,j] > 0.0001:
                        X[i,j] =0.0001
                    if X1[i, j] > 0.0001:
                        X1[i, j] = 0.0001
                    if X2[i,j] > 0.0001:
                        X2[i,j] =0.0001
                    if X[i,j] < 1e-5:
                        X[i, j] = 1e-5
                    if X1[i, j] < 1e-5:
                        X1[i, j] = 1e-5
                    if X2[i, j] < 1e-5:
                        X2[i, j] = 1e-5
                if j ==2:
                    if X[i,j] > 5e-2:
                        X[i,j] = 5e-2
                    if X1[i,j] > 5e-2:
                        X1[i,j] = 5e-2
                    if X2[i, j] > 5e-2:
                        X2[i, j] = 5e-2
                    if X[i,j] < 1e-2:
                        X[i, j] = 1e-2
                    if X1[i,j] < 1e-2:
                        X1[i, j] = 1e-2
                    if X2[i,j] < 1e-2:
                        X2[i, j] = 1e-2
        fit = np.zeros((pop, 1))
        fit1 = np.zeros((pop, 1))
        fit2 = np.zeros((pop, 1))

        for i in range(pop):
            train_bags, train_labels = get_data(posi, nega, channel=4, window_size=151)
            model_type = "CNN"
            loss = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=4,
                                 window_size=151 + 6,
                                 model_file=model_type + '.151', batch_size=batch_size, n_epochs=n_epochs,
                                 num_filters=num_filters, learning_rate=X[i, 0], weight_decay=X[i, 1])
            fit[i, 0] = loss

            train_bags, train_labels = get_data(posi, nega, channel=4, window_size=151)
            model_type = "CNN-LSTM"
            loss1 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=4,
                                  window_size=151 + 6,
                                  model_file=model_type + '.151', batch_size=batch_size, n_epochs=n_epochs,
                                  num_filters=num_filters, learning_rate=X1[i, 0], weight_decay=X1[i, 1])
            fit1[i, 0] = loss1

            train_bags, train_labels = get_data(posi, nega, channel=4, window_size=151)
            model_type = "ResNet"
            loss2 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=4,
                                  window_size=151 + 6,
                                  model_file=model_type + '.151', batch_size=batch_size, n_epochs=n_epochs,
                                  num_filters=num_filters, learning_rate=X2[i, 0], weight_decay=X2[i, 1])
            fit2[i, 0] = loss2
        pFit = fit
        pFit1 = fit1
        pFit2 = fit2
        pX = X
        pX1 = X1
        pX2 = X2
        XX = pX
        XX1 = pX1
        XX2 = pX2
        fMin = np.min(fit[:, 0])
        fMin1 = np.min(fit1[:, 0])
        fMin2 = np.min(fit2[:, 0])
        bestI = np.argmin(fit[:, 0])
        bestI1 = np.argmin(fit1[:, 0])
        bestI2 = np.argmin(fit2[:, 0])
        bestX = X[bestI, :]
        bestX1 = X1[bestI1, :]
        bestX2 = X2[bestI2, :]

        for t in range(M):
            B = np.argmax(pFit[:, 0])
            B1 = np.argmax(pFit1[:, 0])
            B2 = np.argmax(pFit2[:, 0])
            worse = X[B, :]
            worse1 = X1[B1, :]
            worse2 = X2[B2, :]
            r2 = np.random.rand(1)
            for i in range(pNum):
                if r2 < 0.9:
                    a = np.random.rand(1)
                    if a > 0.1:
                        a = 1
                    else:
                        a = -1
                    X[i, :] = pX[i, :] + 0.3 * np.abs(pX[i, :] - worse) + a * 0.1 * (XX[i, :])
                    X1[i, :] = pX1[i, :] + 0.3 * np.abs(pX1[i, :] - worse1) + a * 0.1 * (XX1[i, :])
                    X2[i, :] = pX2[i, :] + 0.3 * np.abs(pX2[i, :] - worse2) + a * 0.1 * (XX2[i, :])
                else:
                    aaa = np.random.randint(180, size=1)
                    if aaa == 0 or aaa == 90 or aaa == 180:
                        X[i, :] = pX[i, :]
                        X1[i, :] = pX1[i, :]
                        X2[i, :] = pX2[i, :]
                    theta = aaa * math.pi / 180
                    X[i, :] = pX[i, :] + math.tan(theta) * np.abs(pX[i, :] - XX[i, :])
                    X1[i, :] = pX1[i, :] + math.tan(theta) * np.abs(pX1[i, :] - XX1[i, :])
                    X2[i, :] = pX2[i, :] + math.tan(theta) * np.abs(pX2[i, :] - XX2[i, :])
                X[i, :] = Bounds(X[i, :], lb, ub)
                X1[i, :] = Bounds(X1[i, :], lb, ub)
                X2[i, :] = Bounds(X2[i, :], lb, ub)
                train_bags, train_labels = get_data(posi, nega, channel=4, window_size=151)
                model_type = "CNN"
                loss = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=4,
                                     window_size=151 + 6,
                                     model_file=model_type + '.151', batch_size=batch_size, n_epochs=n_epochs,
                                     num_filters=num_filters, learning_rate=X[i, 0], weight_decay=X[i, 1])
                fit[i, 0] = loss

                train_bags, train_labels = get_data(posi, nega, channel=4, window_size=151)
                model_type = "CNN-LSTM"
                loss1 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=4,
                                      window_size=151 + 6,
                                      model_file=model_type + '.151', batch_size=batch_size, n_epochs=n_epochs,
                                      num_filters=num_filters, learning_rate=X1[i, 0], weight_decay=X1[i, 1])
                fit1[i, 0] = loss1

                train_bags, train_labels = get_data(posi, nega, channel=4, window_size=151)
                model_type = "ResNet"
                loss2 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=4,
                                      window_size=151 + 6,
                                      model_file=model_type + '.151', batch_size=batch_size, n_epochs=n_epochs,
                                      num_filters=num_filters, learning_rate=X2[i, 0], weight_decay=X2[i, 1])
                fit2[i, 0] = loss2
            bestII = np.argmin(fit[:, 0])
            bestII1 = np.argmin(fit1[:, 0])
            bestII2 = np.argmin(fit2[:, 0])
            bestXX = X[bestII, :]
            bestXX1 = X1[bestII1, :]
            bestXX2 = X2[bestII2, :]
            R = 1 - t / M
            Xnew1 = bestXX * (1 - R)
            Xnew2 = bestXX * (1 + R)
            Xnew1l = bestXX1 * (1 - R)
            Xnew2l = bestXX1 * (1 + R)
            Xnew1r = bestXX2 * (1 - R)
            Xnew2r = bestXX2 * (1 + R)
            Xnew1 = Bounds(Xnew1, lb, ub)
            Xnew2 = Bounds(Xnew2, lb, ub)
            Xnew1l = Bounds(Xnew1l, lb, ub)
            Xnew2l = Bounds(Xnew2l, lb, ub)
            Xnew1r = Bounds(Xnew1r, lb, ub)
            Xnew2r = Bounds(Xnew2r, lb, ub)
            Xnew11 = bestX * (1 - R)
            Xnew22 = bestX * (1 + R)
            Xnew11l = bestX1 * (1 - R)
            Xnew22l = bestX1 * (1 + R)
            Xnew11r = bestX2 * (1 - R)
            Xnew22r = bestX2 * (1 + R)
            Xnew11 = Bounds(Xnew11, lb, ub)
            Xnew22 = Bounds(Xnew22, lb, ub)
            Xnew11l = Bounds(Xnew11l, lb, ub)
            Xnew22l = Bounds(Xnew22l, lb, ub)
            Xnew11r = Bounds(Xnew11r, lb, ub)
            Xnew22r = Bounds(Xnew22r, lb, ub)
            xLB = swapfun(Xnew1)
            xUB = swapfun(Xnew2)
            xLB1 = swapfun(Xnew1l)
            xUB1 = swapfun(Xnew2l)
            xLB2 = swapfun(Xnew1r)
            xUB2 = swapfun(Xnew2r)
            for i in range(pNum + 1, 12):
                X[i, :] = bestXX + (np.random.rand(1, dim)) * (pX[i, :] - Xnew1) + (np.random.rand(1, dim)) * (
                        pX[i, :] - Xnew2)
                X1[i, :] = bestXX1 + (np.random.rand(1, dim)) * (pX1[i, :] - Xnew1l) + (np.random.rand(1, dim)) * (
                        pX1[i, :] - Xnew2l)
                X2[i, :] = bestXX2 + (np.random.rand(1, dim)) * (pX2[i, :] - Xnew1r) + (np.random.rand(1, dim)) * (
                        pX2[i, :] - Xnew2r)
                X[i, :] = Bounds(X[i, :], xLB, xUB)
                X1[i, :] = Bounds(X1[i, :], xLB1, xUB1)
                X2[i, :] = Bounds(X2[i, :], xLB2, xUB2)
                train_bags, train_labels = get_data(posi, nega, channel=4, window_size=151)
                model_type = "CNN"
                loss = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=4,
                                     window_size=151 + 6,
                                     model_file=model_type + '.151', batch_size=batch_size, n_epochs=n_epochs,
                                     num_filters=num_filters, learning_rate=X[i, 0], weight_decay=X[i, 1])
                fit[i, 0] = loss

                train_bags, train_labels = get_data(posi, nega, channel=4, window_size=151)
                model_type = "CNN-LSTM"
                loss1 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=4,
                                      window_size=151 + 6,
                                      model_file=model_type + '.151', batch_size=batch_size, n_epochs=n_epochs,
                                      num_filters=num_filters, learning_rate=X1[i, 0], weight_decay=X1[i, 1])
                fit1[i, 0] = loss1

                train_bags, train_labels = get_data(posi, nega, channel=4, window_size=151)
                model_type = "ResNet"
                loss2 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=4,
                                      window_size=151 + 6,
                                      model_file=model_type + '.151', batch_size=batch_size, n_epochs=n_epochs,
                                      num_filters=num_filters, learning_rate=X2[i, 0], weight_decay=X2[i, 1])
                fit2[i, 0] = loss2
            for i in range(13, 19):
                X[i, :] = pX[i, :] + ((np.random.randn(1)) * (pX[i, :] - Xnew11) + (
                        (np.random.rand(1, dim)) * (pX[i, :] - Xnew22)))
                X1[i, :] = pX1[i, :] + ((np.random.randn(1)) * (pX1[i, :] - Xnew11l) + (
                        (np.random.rand(1, dim)) * (pX1[i, :] - Xnew22l)))
                X2[i, :] = pX2[i, :] + ((np.random.randn(1)) * (pX2[i, :] - Xnew11r) + (
                        (np.random.rand(1, dim)) * (pX2[i, :] - Xnew22r)))
                X[i, :] = Bounds(X[i, :], lb, ub)
                X1[i, :] = Bounds(X1[i, :], lb, ub)
                X2[i, :] = Bounds(X2[i, :], lb, ub)
                train_bags, train_labels = get_data(posi, nega, channel=4, window_size=151)
                model_type = "CNN"
                loss = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=4,
                                     window_size=151 + 6,
                                     model_file=model_type + '.151', batch_size=batch_size, n_epochs=n_epochs,
                                     num_filters=num_filters, learning_rate=X[i, 0], weight_decay=X[i, 1])
                fit[i, 0] = loss

                train_bags, train_labels = get_data(posi, nega, channel=4, window_size=151)
                model_type = "CNN-LSTM"
                loss1 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=4,
                                      window_size=151 + 6,
                                      model_file=model_type + '.151', batch_size=batch_size, n_epochs=n_epochs,
                                      num_filters=num_filters, learning_rate=X1[i, 0], weight_decay=X1[i, 1])
                fit1[i, 0] = loss1

                train_bags, train_labels = get_data(posi, nega, channel=4, window_size=151)
                model_type = "ResNet"
                loss2 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=4,
                                      window_size=151 + 6,
                                      model_file=model_type + '.151', batch_size=batch_size, n_epochs=n_epochs,
                                      num_filters=num_filters, learning_rate=X2[i, 0], weight_decay=X2[i, 1])
                fit2[i, 0] = loss2
            for j in range(20, pop):
                X[j, :] = levy(dim)*bestX + np.random.randn(1, dim) * (np.abs(pX[j, :] - bestXX) + np.abs(pX[j, :] - bestX)) / 2
                X1[j, :] = levy(dim)*bestX1 + np.random.randn(1, dim) * (
                        np.abs(pX1[j, :] - bestXX1) + np.abs(pX1[j, :] - bestX1)) / 2
                X2[j, :] = levy(dim)*bestX2 + np.random.randn(1, dim) * (
                        np.abs(pX2[j, :] - bestXX2) + np.abs(pX2[j, :] - bestX2)) / 2
                X[j, :] = Bounds(X[j, :], lb, ub)
                X1[j, :] = Bounds(X1[j, :], lb, ub)
                X2[j, :] = Bounds(X2[j, :], lb, ub)
                train_bags, train_labels = get_data(posi, nega, channel=4, window_size=151)
                model_type = "CNN"
                loss = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=4,
                                     window_size=151 + 6,
                                     model_file=model_type + '.151', batch_size=batch_size, n_epochs=n_epochs,
                                     num_filters=num_filters, learning_rate=X[j, 0], weight_decay=X[j, 1])
                fit[j, 0] = loss
                train_bags, train_labels = get_data(posi, nega, channel=4, window_size=151)
                model_type = "CNN-LSTM"
                loss1 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=4,
                                      window_size=151 + 6,
                                      model_file=model_type + '.151', batch_size=batch_size, n_epochs=n_epochs,
                                      num_filters=num_filters, learning_rate=X1[j, 0], weight_decay=X1[j, 1])
                fit1[j, 0] = loss1
                train_bags, train_labels = get_data(posi, nega, channel=4, window_size=151)
                model_type = "ResNet"
                loss2 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=4,
                                      window_size=151 + 6,
                                      model_file=model_type + '.151', batch_size=batch_size, n_epochs=n_epochs,
                                      num_filters=num_filters, learning_rate=X2[j, 0], weight_decay=X2[j, 1])
                fit2[j, 0] = loss2
            XX = pX
            XX1 = pX1
            XX2 = pX2
            for i in range(pop):
                if fit[i, 0] < pFit[i, 0]:
                    pFit[i, 0] = fit[i, 0]
                    pX[i, :] = X[i, :]
                if pFit[i, 0] < fMin:
                    fMin = pFit[i, 0]
                    bestX = pX[i, :]
            for i in range(pop):
                if fit1[i, 0] < pFit1[i, 0]:
                    pFit1[i, 0] = fit1[i, 0]
                    pX1[i, :] = X1[i, :]
                if pFit1[i, 0] < fMin1:
                    fMin1 = pFit1[i, 0]
                    bestX1 = pX1[i, :]
            for i in range(pop):
                if fit2[i, 0] < pFit2[i, 0]:
                    pFit2[i, 0] = fit2[i, 0]
                    pX2[i, :] = X2[i, :]
                if pFit2[i, 0] < fMin2:
                    fMin2 = pFit2[i, 0]
                    bestX2 = pX2[i, :]
            X = pX
            X1 = pX1
            X2 = pX2

        print("151")
        train_bags, train_labels = get_data(posi, nega, channel = 4, window_size = 151)
        model_type = "CNN"
        train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 4, window_size = 151 + 6, model_file = model_type + '.151', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters, learning_rate=bestX[0], weight_decay=bestX[1])
        model_type = "CNN-LSTM"
        train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 4, window_size = 151 + 6, model_file = model_type + '.151', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters, learning_rate=bestX1[0], weight_decay=bestX1[1])
        model_type = "ResNet"
        train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 4, window_size = 151 + 6, model_file = model_type + '.151', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters, learning_rate=bestX2[0], weight_decay=bestX2[1])

        pop = 30
        M = 1
        c = [0.0001, 1e-5]
        d = [5e-2, 1e-2]
        dim = 2
        P_percent = 0.2
        pNum = round(pop * P_percent)
        lb = c * np.ones((1, dim))
        ub = d * np.ones((1, dim))
        X = np.zeros((pop, dim))
        X1 = np.zeros((pop, dim))
        X2 = np.zeros((pop, dim))
        for i in range(pop):
            for j in range(dim):
                X[i, j] = X[i, j] + 1e-5 - np.mod(2.2 * np.sin(2 * np.pi * X[i, j]) / (2 * np.pi), 1)
                X1[i, j] = X1[i, j] + 1e-5 - np.mod(2.2 * np.sin(2 * np.pi * X1[i, j]) / (2 * np.pi), 1)
                X2[i, j] = X2[i, j] + 1e-5 - np.mod(2.2 * np.sin(2 * np.pi * X2[i, j]) / (2 * np.pi), 1)
                if j == 1:
                    if X[i, j] > 0.0001:
                        X[i, j] = 0.0001
                    if X1[i, j] > 0.0001:
                        X1[i, j] = 0.0001
                    if X2[i, j] > 0.0001:
                        X2[i, j] = 0.0001
                    if X[i, j] < 1e-5:
                        X[i, j] = 1e-5
                    if X1[i, j] < 1e-5:
                        X1[i, j] = 1e-5
                    if X2[i, j] < 1e-5:
                        X2[i, j] = 1e-5
                if j == 2:
                    if X[i, j] > 5e-2:
                        X[i, j] = 5e-2
                    if X1[i, j] > 5e-2:
                        X1[i, j] = 5e-2
                    if X2[i, j] > 5e-2:
                        X2[i, j] = 5e-2
                    if X[i, j] < 1e-2:
                        X[i, j] = 1e-2
                    if X1[i, j] < 1e-2:
                        X1[i, j] = 1e-2
                    if X2[i, j] < 1e-2:
                        X2[i, j] = 1e-2
        fit = np.zeros((pop, 1))
        fit1 = np.zeros((pop, 1))
        fit2 = np.zeros((pop, 1))

        for i in range(pop):
            train_bags, train_labels = get_data(posi, nega, channel=3, window_size=201)
            model_type = "CNN"
            loss = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=3,
                                 window_size=201 + 6,
                                 model_file=model_type + '.201', batch_size=batch_size, n_epochs=n_epochs,
                                 num_filters=num_filters, learning_rate=X[i, 0], weight_decay=X[i, 1])
            fit[i, 0] = loss
            train_bags, train_labels = get_data(posi, nega, channel=3, window_size=201)
            model_type = "CNN-LSTM"
            loss1 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=3,
                                  window_size=201 + 6,
                                  model_file=model_type + '.201', batch_size=batch_size, n_epochs=n_epochs,
                                  num_filters=num_filters, learning_rate=X1[i, 0], weight_decay=X1[i, 1])
            fit1[i, 0] = loss1
            train_bags, train_labels = get_data(posi, nega, channel=3, window_size=201)
            model_type = "ResNet"
            loss2 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=3,
                                  window_size=201 + 6,
                                  model_file=model_type + '.201', batch_size=batch_size, n_epochs=n_epochs,
                                  num_filters=num_filters, learning_rate=X2[i, 0], weight_decay=X2[i, 1])
            fit2[i, 0] = loss2
        pFit = fit
        pFit1 = fit1
        pFit2 = fit2
        pX = X
        pX1 = X1
        pX2 = X2
        XX = pX
        XX1 = pX1
        XX2 = pX2
        fMin = np.min(fit[:, 0])
        fMin1 = np.min(fit1[:, 0])
        fMin2 = np.min(fit2[:, 0])
        bestI = np.argmin(fit[:, 0])
        bestI1 = np.argmin(fit1[:, 0])
        bestI2 = np.argmin(fit2[:, 0])
        bestX = X[bestI, :]
        bestX1 = X1[bestI1, :]
        bestX2 = X2[bestI2, :]

        for t in range(M):
            B = np.argmax(pFit[:, 0])
            B1 = np.argmax(pFit1[:, 0])
            B2 = np.argmax(pFit2[:, 0])
            worse = X[B, :]
            worse1 = X1[B1, :]
            worse2 = X2[B2, :]
            r2 = np.random.rand(1)
            for i in range(pNum):
                if r2 < 0.9:
                    a = np.random.rand(1)
                    if a > 0.1:
                        a = 1
                    else:
                        a = -1
                    X[i, :] = pX[i, :] + 0.3 * np.abs(pX[i, :] - worse) + a * 0.1 * (XX[i, :])
                    X1[i, :] = pX1[i, :] + 0.3 * np.abs(pX1[i, :] - worse1) + a * 0.1 * (XX1[i, :])
                    X2[i, :] = pX2[i, :] + 0.3 * np.abs(pX2[i, :] - worse2) + a * 0.1 * (XX2[i, :])
                else:
                    aaa = np.random.randint(180, size=1)
                    if aaa == 0 or aaa == 90 or aaa == 180:
                        X[i, :] = pX[i, :]
                        X1[i, :] = pX1[i, :]
                        X2[i, :] = pX2[i, :]
                    theta = aaa * math.pi / 180
                    X[i, :] = pX[i, :] + math.tan(theta) * np.abs(pX[i, :] - XX[i, :])
                    X1[i, :] = pX1[i, :] + math.tan(theta) * np.abs(pX1[i, :] - XX1[i, :])
                    X2[i, :] = pX2[i, :] + math.tan(theta) * np.abs(pX2[i, :] - XX2[i, :])
                X[i, :] = Bounds(X[i, :], lb, ub)
                X1[i, :] = Bounds(X1[i, :], lb, ub)
                X2[i, :] = Bounds(X2[i, :], lb, ub)
                train_bags, train_labels = get_data(posi, nega, channel=3, window_size=201)
                model_type = "CNN"
                loss = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=3,
                                     window_size=201 + 6,
                                     model_file=model_type + '.201', batch_size=batch_size, n_epochs=n_epochs,
                                     num_filters=num_filters, learning_rate=X[i, 0], weight_decay=X[i, 1])
                fit[i, 0] = loss
                train_bags, train_labels = get_data(posi, nega, channel=3, window_size=201)
                model_type = "CNN-LSTM"
                loss1 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=3,
                                      window_size=201 + 6,
                                      model_file=model_type + '.201', batch_size=batch_size, n_epochs=n_epochs,
                                      num_filters=num_filters, learning_rate=X1[i, 0], weight_decay=X1[i, 1])
                fit1[i, 0] = loss1
                train_bags, train_labels = get_data(posi, nega, channel=3, window_size=201)
                model_type = "ResNet"
                loss2 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=3,
                                      window_size=201 + 6,
                                      model_file=model_type + '.201', batch_size=batch_size, n_epochs=n_epochs,
                                      num_filters=num_filters, learning_rate=X2[i, 0], weight_decay=X2[i, 1])
                fit2[i, 0] = loss2
            bestII = np.argmin(fit[:, 0])
            bestII1 = np.argmin(fit1[:, 0])
            bestII2 = np.argmin(fit2[:, 0])
            bestXX = X[bestII, :]
            bestXX1 = X1[bestII1, :]
            bestXX2 = X2[bestII2, :]
            R = 1 - t / M
            Xnew1 = bestXX * (1 - R)
            Xnew2 = bestXX * (1 + R)
            Xnew1l = bestXX1 * (1 - R)
            Xnew2l = bestXX1 * (1 + R)
            Xnew1r = bestXX2 * (1 - R)
            Xnew2r = bestXX2 * (1 + R)
            Xnew1 = Bounds(Xnew1, lb, ub)
            Xnew2 = Bounds(Xnew2, lb, ub)
            Xnew1l = Bounds(Xnew1l, lb, ub)
            Xnew2l = Bounds(Xnew2l, lb, ub)
            Xnew1r = Bounds(Xnew1r, lb, ub)
            Xnew2r = Bounds(Xnew2r, lb, ub)
            Xnew11 = bestX * (1 - R)
            Xnew22 = bestX * (1 + R)
            Xnew11l = bestX1 * (1 - R)
            Xnew22l = bestX1 * (1 + R)
            Xnew11r = bestX2 * (1 - R)
            Xnew22r = bestX2 * (1 + R)
            Xnew11 = Bounds(Xnew11, lb, ub)
            Xnew22 = Bounds(Xnew22, lb, ub)
            Xnew11l = Bounds(Xnew11l, lb, ub)
            Xnew22l = Bounds(Xnew22l, lb, ub)
            Xnew11r = Bounds(Xnew11r, lb, ub)
            Xnew22r = Bounds(Xnew22r, lb, ub)
            xLB = swapfun(Xnew1)
            xUB = swapfun(Xnew2)
            xLB1 = swapfun(Xnew1l)
            xUB1 = swapfun(Xnew2l)
            xLB2 = swapfun(Xnew1r)
            xUB2 = swapfun(Xnew2r)
            for i in range(pNum + 1, 12):
                X[i, :] = bestXX + (np.random.rand(1, dim)) * (pX[i, :] - Xnew1) + (np.random.rand(1, dim)) * (
                        pX[i, :] - Xnew2)
                X1[i, :] = bestXX1 + (np.random.rand(1, dim)) * (pX1[i, :] - Xnew1l) + (np.random.rand(1, dim)) * (
                        pX1[i, :] - Xnew2l)
                X2[i, :] = bestXX2 + (np.random.rand(1, dim)) * (pX2[i, :] - Xnew1r) + (np.random.rand(1, dim)) * (
                        pX2[i, :] - Xnew2r)
                X[i, :] = Bounds(X[i, :], xLB, xUB)
                X1[i, :] = Bounds(X1[i, :], xLB1, xUB1)
                X2[i, :] = Bounds(X2[i, :], xLB2, xUB2)
                train_bags, train_labels = get_data(posi, nega, channel=3, window_size=201)
                model_type = "CNN"
                loss = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=3,
                                     window_size=201 + 6,
                                     model_file=model_type + '.201', batch_size=batch_size, n_epochs=n_epochs,
                                     num_filters=num_filters, learning_rate=X[i, 0], weight_decay=X[i, 1])
                fit[i, 0] = loss

                train_bags, train_labels = get_data(posi, nega, channel=3, window_size=201)
                model_type = "CNN-LSTM"
                loss1 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=3,
                                      window_size=201 + 6,
                                      model_file=model_type + '.201', batch_size=batch_size, n_epochs=n_epochs,
                                      num_filters=num_filters, learning_rate=X1[i, 0], weight_decay=X1[i, 1])
                fit1[i, 0] = loss1

                train_bags, train_labels = get_data(posi, nega, channel=3, window_size=201)
                model_type = "ResNet"
                loss2 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=3,
                                      window_size=201 + 6,
                                      model_file=model_type + '.201', batch_size=batch_size, n_epochs=n_epochs,
                                      num_filters=num_filters, learning_rate=X2[i, 0], weight_decay=X2[i, 1])
                fit2[i, 0] = loss2
            for i in range(13, 19):
                X[i, :] = pX[i, :] + ((np.random.randn(1)) * (pX[i, :] - Xnew11) + (
                        (np.random.rand(1, dim)) * (pX[i, :] - Xnew22)))
                X1[i, :] = pX1[i, :] + ((np.random.randn(1)) * (pX1[i, :] - Xnew11l) + (
                        (np.random.rand(1, dim)) * (pX1[i, :] - Xnew22l)))
                X2[i, :] = pX2[i, :] + ((np.random.randn(1)) * (pX2[i, :] - Xnew11r) + (
                        (np.random.rand(1, dim)) * (pX2[i, :] - Xnew22r)))
                X[i, :] = Bounds(X[i, :], lb, ub)
                X1[i, :] = Bounds(X1[i, :], lb, ub)
                X2[i, :] = Bounds(X2[i, :], lb, ub)
                train_bags, train_labels = get_data(posi, nega, channel=3, window_size=201)
                model_type = "CNN"
                loss = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=3,
                                     window_size=201 + 6,
                                     model_file=model_type + '.201', batch_size=batch_size, n_epochs=n_epochs,
                                     num_filters=num_filters, learning_rate=X[i, 0], weight_decay=X[i, 1])
                fit[i, 0] = loss

                train_bags, train_labels = get_data(posi, nega, channel=3, window_size=201)
                model_type = "CNN-LSTM"
                loss1 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=3,
                                      window_size=201 + 6,
                                      model_file=model_type + '.201', batch_size=batch_size, n_epochs=n_epochs,
                                      num_filters=num_filters, learning_rate=X1[i, 0], weight_decay=X1[i, 1])
                fit1[i, 0] = loss1

                train_bags, train_labels = get_data(posi, nega, channel=3, window_size=201)
                model_type = "ResNet"
                loss2 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=3,
                                      window_size=201 + 6,
                                      model_file=model_type + '.201', batch_size=batch_size, n_epochs=n_epochs,
                                      num_filters=num_filters, learning_rate=X2[i, 0], weight_decay=X2[i, 1])
                fit2[i, 0] = loss2
            for j in range(20, pop):
                X[j, :] = levy(dim)*bestX + np.random.randn(1, dim) * (np.abs(pX[j, :] - bestXX) + np.abs(pX[j, :] - bestX)) / 2
                X1[j, :] = levy(dim)*bestX1 + np.random.randn(1, dim) * (
                        np.abs(pX1[j, :] - bestXX1) + np.abs(pX1[j, :] - bestX1)) / 2
                X2[j, :] = levy(dim)*bestX2 + np.random.randn(1, dim) * (
                        np.abs(pX2[j, :] - bestXX2) + np.abs(pX2[j, :] - bestX2)) / 2
                X[j, :] = Bounds(X[j, :], lb, ub)
                X1[j, :] = Bounds(X1[j, :], lb, ub)
                X2[j, :] = Bounds(X2[j, :], lb, ub)
                train_bags, train_labels = get_data(posi, nega, channel=3, window_size=201)
                model_type = "CNN"
                loss = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=3,
                                     window_size=201 + 6,
                                     model_file=model_type + '.201', batch_size=batch_size, n_epochs=n_epochs,
                                     num_filters=num_filters, learning_rate=X[j, 0], weight_decay=X[j, 1])
                fit[j, 0] = loss
                train_bags, train_labels = get_data(posi, nega, channel=3, window_size=201)
                model_type = "CNN-LSTM"
                loss1 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=3,
                                      window_size=201 + 6,
                                      model_file=model_type + '.201', batch_size=batch_size, n_epochs=n_epochs,
                                      num_filters=num_filters, learning_rate=X1[j, 0], weight_decay=X1[j, 1])
                fit1[j, 0] = loss1
                train_bags, train_labels = get_data(posi, nega, channel=3, window_size=201)
                model_type = "ResNet"
                loss2 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=3,
                                      window_size=201 + 6,
                                      model_file=model_type + '.201', batch_size=batch_size, n_epochs=n_epochs,
                                      num_filters=num_filters, learning_rate=X2[j, 0], weight_decay=X2[j, 1])
                fit2[j, 0] = loss2
            XX = pX
            XX1 = pX1
            XX2 = pX2
            for i in range(pop):
                if fit[i, 0] < pFit[i, 0]:
                    pFit[i, 0] = fit[i, 0]
                    pX[i, :] = X[i, :]
                if pFit[i, 0] < fMin:
                    fMin = pFit[i, 0]
                    bestX = pX[i, :]
            for i in range(pop):
                if fit1[i, 0] < pFit1[i, 0]:
                    pFit1[i, 0] = fit1[i, 0]
                    pX1[i, :] = X1[i, :]
                if pFit1[i, 0] < fMin1:
                    fMin1 = pFit1[i, 0]
                    bestX1 = pX1[i, :]
            for i in range(pop):
                if fit2[i, 0] < pFit2[i, 0]:
                    pFit2[i, 0] = fit2[i, 0]
                    pX2[i, :] = X2[i, :]
                if pFit2[i, 0] < fMin2:
                    fMin2 = pFit2[i, 0]
                    bestX2 = pX2[i, :]
            X = pX
            X1 = pX1
            X2 = pX2

        print("201")
        train_bags, train_labels = get_data(posi, nega, channel = 3, window_size = 201)
        model_type = "CNN"
        train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 3, window_size = 201 + 6, model_file = model_type + '.201', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters, learning_rate=bestX[0], weight_decay=bestX[1])
        model_type = "CNN-LSTM"
        train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 3, window_size = 201 + 6, model_file = model_type + '.201', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters, learning_rate=bestX1[0], weight_decay=bestX1[1])
        model_type = "ResNet"
        train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 3, window_size = 201 + 6, model_file = model_type + '.201', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters, learning_rate=bestX2[0], weight_decay=bestX2[1])

        pop = 30
        M = 1
        c = [0.0001, 1e-5]
        d = [5e-2, 1e-2]
        dim = 2
        P_percent = 0.2
        pNum = round(pop * P_percent)
        lb = c * np.ones((1, dim))
        ub = d * np.ones((1, dim))
        X = np.zeros((pop, dim))
        X1 = np.zeros((pop, dim))
        X2 = np.zeros((pop, dim))
        for i in range(pop):
            for j in range(dim):
                X[i, j] = X[i, j] + 1e-5 - np.mod(2.2 * np.sin(2 * np.pi * X[i, j]) / (2 * np.pi), 1)
                X1[i, j] = X1[i, j] + 1e-5 - np.mod(2.2 * np.sin(2 * np.pi * X1[i, j]) / (2 * np.pi), 1)
                X2[i, j] = X2[i, j] + 1e-5 - np.mod(2.2 * np.sin(2 * np.pi * X2[i, j]) / (2 * np.pi), 1)
                if j == 1:
                    if X[i, j] > 0.0001:
                        X[i, j] = 0.0001
                    if X1[i, j] > 0.0001:
                        X1[i, j] = 0.0001
                    if X2[i, j] > 0.0001:
                        X2[i, j] = 0.0001
                    if X[i, j] < 1e-5:
                        X[i, j] = 1e-5
                    if X1[i, j] < 1e-5:
                        X1[i, j] = 1e-5
                    if X2[i, j] < 1e-5:
                        X2[i, j] = 1e-5
                if j == 2:
                    if X[i, j] > 5e-2:
                        X[i, j] = 5e-2
                    if X1[i, j] > 5e-2:
                        X1[i, j] = 5e-2
                    if X2[i, j] > 5e-2:
                        X2[i, j] = 5e-2
                    if X[i, j] < 1e-2:
                        X[i, j] = 1e-2
                    if X1[i, j] < 1e-2:
                        X1[i, j] = 1e-2
                    if X2[i, j] < 1e-2:
                        X2[i, j] = 1e-2
        fit = np.zeros((pop, 1))
        fit1 = np.zeros((pop, 1))
        fit2 = np.zeros((pop, 1))

        for i in range(pop):

            train_bags, train_labels = get_data(posi, nega, channel=2, window_size=251)
            model_type = "CNN"
            loss = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=2,
                                 window_size=251 + 6,
                                 model_file=model_type + '.251', batch_size=batch_size, n_epochs=n_epochs,
                                 num_filters=num_filters, learning_rate=X[i, 0], weight_decay=X[i, 1])
            fit[i, 0] = loss

            train_bags, train_labels = get_data(posi, nega, channel=2, window_size=251)
            model_type = "CNN-LSTM"
            loss1 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=2,
                                  window_size=251 + 6,
                                  model_file=model_type + '.251', batch_size=batch_size, n_epochs=n_epochs,
                                  num_filters=num_filters, learning_rate=X1[i, 0], weight_decay=X1[i, 1])
            fit1[i, 0] = loss1

            train_bags, train_labels = get_data(posi, nega, channel=2, window_size=251)
            model_type = "ResNet"
            loss2 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=2,
                                  window_size=251 + 6,
                                  model_file=model_type + '.251', batch_size=batch_size, n_epochs=n_epochs,
                                  num_filters=num_filters, learning_rate=X2[i, 0], weight_decay=X2[i, 1])
            fit2[i, 0] = loss2
        pFit = fit
        pFit1 = fit1
        pFit2 = fit2
        pX = X
        pX1 = X1
        pX2 = X2
        XX = pX
        XX1 = pX1
        XX2 = pX2
        fMin = np.min(fit[:, 0])
        fMin1 = np.min(fit1[:, 0])
        fMin2 = np.min(fit2[:, 0])
        bestI = np.argmin(fit[:, 0])
        bestI1 = np.argmin(fit1[:, 0])
        bestI2 = np.argmin(fit2[:, 0])
        bestX = X[bestI, :]
        bestX1 = X1[bestI1, :]
        bestX2 = X2[bestI2, :]

        for t in range(M):
            B = np.argmax(pFit[:, 0])
            B1 = np.argmax(pFit1[:, 0])
            B2 = np.argmax(pFit2[:, 0])
            worse = X[B, :]
            worse1 = X1[B1, :]
            worse2 = X2[B2, :]
            r2 = np.random.rand(1)
            for i in range(pNum):
                if r2 < 0.9:
                    a = np.random.rand(1)
                    if a > 0.1:
                        a = 1
                    else:
                        a = -1
                    X[i, :] = pX[i, :] + 0.3 * np.abs(pX[i, :] - worse) + a * 0.1 * (XX[i, :])
                    X1[i, :] = pX1[i, :] + 0.3 * np.abs(pX1[i, :] - worse1) + a * 0.1 * (XX1[i, :])
                    X2[i, :] = pX2[i, :] + 0.3 * np.abs(pX2[i, :] - worse2) + a * 0.1 * (XX2[i, :])
                else:
                    aaa = np.random.randint(180, size=1)
                    if aaa == 0 or aaa == 90 or aaa == 180:
                        X[i, :] = pX[i, :]
                        X1[i, :] = pX1[i, :]
                        X2[i, :] = pX2[i, :]
                    theta = aaa * math.pi / 180
                    X[i, :] = pX[i, :] + math.tan(theta) * np.abs(pX[i, :] - XX[i, :])
                    X1[i, :] = pX1[i, :] + math.tan(theta) * np.abs(pX1[i, :] - XX1[i, :])
                    X2[i, :] = pX2[i, :] + math.tan(theta) * np.abs(pX2[i, :] - XX2[i, :])
                X[i, :] = Bounds(X[i, :], lb, ub)
                X1[i, :] = Bounds(X1[i, :], lb, ub)
                X2[i, :] = Bounds(X2[i, :], lb, ub)
                train_bags, train_labels = get_data(posi, nega, channel=2, window_size=251)
                model_type = "CNN"
                loss = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=2,
                                     window_size=251 + 6,
                                     model_file=model_type + '.251', batch_size=batch_size, n_epochs=n_epochs,
                                     num_filters=num_filters, learning_rate=X[i, 0], weight_decay=X[i, 1])
                fit[i, 0] = loss

                train_bags, train_labels = get_data(posi, nega, channel=2, window_size=251)
                model_type = "CNN-LSTM"
                loss1 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=2,
                                      window_size=251 + 6,
                                      model_file=model_type + '.251', batch_size=batch_size, n_epochs=n_epochs,
                                      num_filters=num_filters, learning_rate=X1[i, 0], weight_decay=X1[i, 1])
                fit1[i, 0] = loss1

                train_bags, train_labels = get_data(posi, nega, channel=2, window_size=251)
                model_type = "ResNet"
                loss2 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=2,
                                      window_size=251 + 6,
                                      model_file=model_type + '.251', batch_size=batch_size, n_epochs=n_epochs,
                                      num_filters=num_filters, learning_rate=X2[i, 0], weight_decay=X2[i, 1])
                fit2[i, 0] = loss2
            bestII = np.argmin(fit[:, 0])
            bestII1 = np.argmin(fit1[:, 0])
            bestII2 = np.argmin(fit2[:, 0])
            bestXX = X[bestII, :]
            bestXX1 = X1[bestII1, :]
            bestXX2 = X2[bestII2, :]
            R = 1 - t / M
            Xnew1 = bestXX * (1 - R)
            Xnew2 = bestXX * (1 + R)
            Xnew1l = bestXX1 * (1 - R)
            Xnew2l = bestXX1 * (1 + R)
            Xnew1r = bestXX2 * (1 - R)
            Xnew2r = bestXX2 * (1 + R)
            Xnew1 = Bounds(Xnew1, lb, ub)
            Xnew2 = Bounds(Xnew2, lb, ub)
            Xnew1l = Bounds(Xnew1l, lb, ub)
            Xnew2l = Bounds(Xnew2l, lb, ub)
            Xnew1r = Bounds(Xnew1r, lb, ub)
            Xnew2r = Bounds(Xnew2r, lb, ub)
            Xnew11 = bestX * (1 - R)
            Xnew22 = bestX * (1 + R)
            Xnew11l = bestX1 * (1 - R)
            Xnew22l = bestX1 * (1 + R)
            Xnew11r = bestX2 * (1 - R)
            Xnew22r = bestX2 * (1 + R)
            Xnew11 = Bounds(Xnew11, lb, ub)
            Xnew22 = Bounds(Xnew22, lb, ub)
            Xnew11l = Bounds(Xnew11l, lb, ub)
            Xnew22l = Bounds(Xnew22l, lb, ub)
            Xnew11r = Bounds(Xnew11r, lb, ub)
            Xnew22r = Bounds(Xnew22r, lb, ub)
            xLB = swapfun(Xnew1)
            xUB = swapfun(Xnew2)
            xLB1 = swapfun(Xnew1l)
            xUB1 = swapfun(Xnew2l)
            xLB2 = swapfun(Xnew1r)
            xUB2 = swapfun(Xnew2r)
            for i in range(pNum + 1, 12):
                X[i, :] = bestXX + (np.random.rand(1, dim)) * (pX[i, :] - Xnew1) + (np.random.rand(1, dim)) * (
                        pX[i, :] - Xnew2)
                X1[i, :] = bestXX1 + (np.random.rand(1, dim)) * (pX1[i, :] - Xnew1l) + (np.random.rand(1, dim)) * (
                        pX1[i, :] - Xnew2l)
                X2[i, :] = bestXX2 + (np.random.rand(1, dim)) * (pX2[i, :] - Xnew1r) + (np.random.rand(1, dim)) * (
                        pX2[i, :] - Xnew2r)
                X[i, :] = Bounds(X[i, :], xLB, xUB)
                X1[i, :] = Bounds(X1[i, :], xLB1, xUB1)
                X2[i, :] = Bounds(X2[i, :], xLB2, xUB2)
                train_bags, train_labels = get_data(posi, nega, channel=2, window_size=251)
                model_type = "CNN"
                loss = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=2,
                                     window_size=251 + 6,
                                     model_file=model_type + '.251', batch_size=batch_size, n_epochs=n_epochs,
                                     num_filters=num_filters, learning_rate=X[i, 0], weight_decay=X[i, 1])
                fit[i, 0] = loss

                train_bags, train_labels = get_data(posi, nega, channel=2, window_size=251)
                model_type = "CNN-LSTM"
                loss1 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=2,
                                      window_size=251 + 6,
                                      model_file=model_type + '.251', batch_size=batch_size, n_epochs=n_epochs,
                                      num_filters=num_filters, learning_rate=X1[i, 0], weight_decay=X1[i, 1])
                fit1[i, 0] = loss1

                train_bags, train_labels = get_data(posi, nega, channel=2, window_size=251)
                model_type = "ResNet"
                loss2 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=2,
                                      window_size=251 + 6,
                                      model_file=model_type + '.251', batch_size=batch_size, n_epochs=n_epochs,
                                      num_filters=num_filters, learning_rate=X2[i, 0], weight_decay=X2[i, 1])
                fit2[i, 0] = loss2
            for i in range(13, 19):
                X[i, :] = pX[i, :] + ((np.random.randn(1)) * (pX[i, :] - Xnew11) + (
                        (np.random.rand(1, dim)) * (pX[i, :] - Xnew22)))
                X1[i, :] = pX1[i, :] + ((np.random.randn(1)) * (pX1[i, :] - Xnew11l) + (
                        (np.random.rand(1, dim)) * (pX1[i, :] - Xnew22l)))
                X2[i, :] = pX2[i, :] + ((np.random.randn(1)) * (pX2[i, :] - Xnew11r) + (
                        (np.random.rand(1, dim)) * (pX2[i, :] - Xnew22r)))
                X[i, :] = Bounds(X[i, :], lb, ub)
                X1[i, :] = Bounds(X1[i, :], lb, ub)
                X2[i, :] = Bounds(X2[i, :], lb, ub)
                train_bags, train_labels = get_data(posi, nega, channel=2, window_size=251)
                model_type = "CNN"
                loss = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=2,
                                     window_size=251 + 6,
                                     model_file=model_type + '.251', batch_size=batch_size, n_epochs=n_epochs,
                                     num_filters=num_filters, learning_rate=X[i, 0], weight_decay=X[i, 1])
                fit[i, 0] = loss

                train_bags, train_labels = get_data(posi, nega, channel=2, window_size=251)
                model_type = "CNN-LSTM"
                loss1 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=2,
                                      window_size=251 + 6,
                                      model_file=model_type + '.251', batch_size=batch_size, n_epochs=n_epochs,
                                      num_filters=num_filters, learning_rate=X1[i, 0], weight_decay=X1[i, 1])
                fit1[i, 0] = loss1

                train_bags, train_labels = get_data(posi, nega, channel=2, window_size=251)
                model_type = "ResNet"
                loss2 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=2,
                                      window_size=251 + 6,
                                      model_file=model_type + '.251', batch_size=batch_size, n_epochs=n_epochs,
                                      num_filters=num_filters, learning_rate=X2[i, 0], weight_decay=X2[i, 1])
                fit2[i, 0] = loss2
            for j in range(20, pop):
                X[j, :] = levy(dim)*bestX + np.random.randn(1, dim) * (np.abs(pX[j, :] - bestXX) + np.abs(pX[j, :] - bestX)) / 2
                X1[j, :] = levy(dim)*bestX1 + np.random.randn(1, dim) * (
                        np.abs(pX1[j, :] - bestXX1) + np.abs(pX1[j, :] - bestX1)) / 2
                X2[j, :] = levy(dim)*bestX2 + np.random.randn(1, dim) * (
                        np.abs(pX2[j, :] - bestXX2) + np.abs(pX2[j, :] - bestX2)) / 2
                X[j, :] = Bounds(X[j, :], lb, ub)
                X1[j, :] = Bounds(X1[j, :], lb, ub)
                X2[j, :] = Bounds(X2[j, :], lb, ub)
                train_bags, train_labels = get_data(posi, nega, channel=2, window_size=251)
                model_type = "CNN"
                loss = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=2,
                                     window_size=251 + 6,
                                     model_file=model_type + '.251', batch_size=batch_size, n_epochs=n_epochs,
                                     num_filters=num_filters, learning_rate=X[j, 0], weight_decay=X[j, 1])
                fit[j, 0] = loss
                train_bags, train_labels = get_data(posi, nega, channel=2, window_size=251)
                model_type = "CNN-LSTM"
                loss1 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=2,
                                      window_size=251 + 6,
                                      model_file=model_type + '.251', batch_size=batch_size, n_epochs=n_epochs,
                                      num_filters=num_filters, learning_rate=X1[j, 0], weight_decay=X1[j, 1])
                fit1[j, 0] = loss1
                train_bags, train_labels = get_data(posi, nega, channel=2, window_size=251)
                model_type = "ResNet"
                loss2 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=2,
                                      window_size=251 + 6,
                                      model_file=model_type + '.251', batch_size=batch_size, n_epochs=n_epochs,
                                      num_filters=num_filters, learning_rate=X2[j, 0], weight_decay=X2[j, 1])
                fit2[j, 0] = loss2
            XX = pX
            XX1 = pX1
            XX2 = pX2
            for i in range(pop):
                if fit[i, 0] < pFit[i, 0]:
                    pFit[i, 0] = fit[i, 0]
                    pX[i, :] = X[i, :]
                if pFit[i, 0] < fMin:
                    fMin = pFit[i, 0]
                    bestX = pX[i, :]
            for i in range(pop):
                if fit1[i, 0] < pFit1[i, 0]:
                    pFit1[i, 0] = fit1[i, 0]
                    pX1[i, :] = X1[i, :]
                if pFit1[i, 0] < fMin1:
                    fMin1 = pFit1[i, 0]
                    bestX1 = pX1[i, :]
            for i in range(pop):
                if fit2[i, 0] < pFit2[i, 0]:
                    pFit2[i, 0] = fit2[i, 0]
                    pX2[i, :] = X2[i, :]
                if pFit2[i, 0] < fMin2:
                    fMin2 = pFit2[i, 0]
                    bestX2 = pX2[i, :]
            X = pX
            X1 = pX1
            X2 = pX2

        print("251")
        train_bags, train_labels = get_data(posi, nega, channel = 2, window_size = 251)
        model_type = "CNN"
        train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 2, window_size = 251 + 6, model_file = model_type + '.251', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters, learning_rate=bestX[0], weight_decay=bestX[1])
        model_type = "CNN-LSTM"
        train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 2, window_size = 251 + 6, model_file = model_type + '.251', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters, learning_rate=bestX1[0], weight_decay=bestX1[1])
        model_type = "ResNet"
        train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 2, window_size = 251 + 6, model_file = model_type + '.251', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters,learning_rate=bestX2[0], weight_decay=bestX2[1])


        pop = 30
        M = 1
        c = [0.0001, 1e-5]
        d = [5e-2, 1e-2]
        dim = 2
        P_percent = 0.2
        pNum = round(pop * P_percent)
        lb = c * np.ones((1, dim))
        ub = d * np.ones((1, dim))
        X = np.zeros((pop, dim))
        X1 = np.zeros((pop, dim))
        X2 = np.zeros((pop, dim))
        for i in range(pop):
            for j in range(dim):
                X[i, j] = X[i, j] + 1e-5 - np.mod(2.2 * np.sin(2 * np.pi * X[i, j]) / (2 * np.pi), 1)
                X1[i, j] = X1[i, j] + 1e-5 - np.mod(2.2 * np.sin(2 * np.pi * X1[i, j]) / (2 * np.pi), 1)
                X2[i, j] = X2[i, j] + 1e-5 - np.mod(2.2 * np.sin(2 * np.pi * X2[i, j]) / (2 * np.pi), 1)
                if j == 1:
                    if X[i, j] > 0.0001:
                        X[i, j] = 0.0001
                    if X1[i, j] > 0.0001:
                        X1[i, j] = 0.0001
                    if X2[i, j] > 0.0001:
                        X2[i, j] = 0.0001
                    if X[i, j] < 1e-5:
                        X[i, j] = 1e-5
                    if X1[i, j] < 1e-5:
                        X1[i, j] = 1e-5
                    if X2[i, j] < 1e-5:
                        X2[i, j] = 1e-5
                if j == 2:
                    if X[i, j] > 5e-2:
                        X[i, j] = 5e-2
                    if X1[i, j] > 5e-2:
                        X1[i, j] = 5e-2
                    if X2[i, j] > 5e-2:
                        X2[i, j] = 5e-2
                    if X[i, j] < 1e-2:
                        X[i, j] = 1e-2
                    if X1[i, j] < 1e-2:
                        X1[i, j] = 1e-2
                    if X2[i, j] < 1e-2:
                        X2[i, j] = 1e-2
        fit = np.zeros((pop, 1))
        fit1 = np.zeros((pop, 1))
        fit2 = np.zeros((pop, 1))

        for i in range(pop):
            X[i, :] = lb + (ub - lb) * np.random.rand(1, dim)
            train_bags, train_labels = get_data(posi, nega, channel=2, window_size=301)
            model_type = "CNN"
            loss = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=2,
                                 window_size=301 + 6,
                                 model_file=model_type + '.301', batch_size=batch_size, n_epochs=n_epochs,
                                 num_filters=num_filters, learning_rate=X[i, 0], weight_decay=X[i, 1])
            fit[i, 0] = loss
            X1[i, :] = lb + (ub - lb) * np.random.rand(1, dim)
            train_bags, train_labels = get_data(posi, nega, channel=2, window_size=301)
            model_type = "CNN-LSTM"
            loss1 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=2,
                                  window_size=301 + 6,
                                  model_file=model_type + '.301', batch_size=batch_size, n_epochs=n_epochs,
                                  num_filters=num_filters, learning_rate=X1[i, 0], weight_decay=X1[i, 1])
            fit1[i, 0] = loss1
            X2[i, :] = lb + (ub - lb) * np.random.rand(1, dim)
            train_bags, train_labels = get_data(posi, nega, channel=2, window_size=301)
            model_type = "ResNet"
            loss2 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=2,
                                  window_size=301 + 6,
                                  model_file=model_type + '.301', batch_size=batch_size, n_epochs=n_epochs,
                                  num_filters=num_filters, learning_rate=X2[i, 0], weight_decay=X2[i, 1])
            fit2[i, 0] = loss2
        pFit = fit
        pFit1 = fit1
        pFit2 = fit2
        pX = X
        pX1 = X1
        pX2 = X2
        XX = pX
        XX1 = pX1
        XX2 = pX2
        fMin = np.min(fit[:, 0])
        fMin1 = np.min(fit1[:, 0])
        fMin2 = np.min(fit2[:, 0])
        bestI = np.argmin(fit[:, 0])
        bestI1 = np.argmin(fit1[:, 0])
        bestI2 = np.argmin(fit2[:, 0])
        bestX = X[bestI, :]
        bestX1 = X1[bestI1, :]
        bestX2 = X2[bestI2, :]

        for t in range(M):
            B = np.argmax(pFit[:, 0])
            B1 = np.argmax(pFit1[:, 0])
            B2 = np.argmax(pFit2[:, 0])
            worse = X[B, :]
            worse1 = X1[B1, :]
            worse2 = X2[B2, :]
            r2 = np.random.rand(1)
            for i in range(pNum):
                if r2 < 0.9:
                    a = np.random.rand(1)
                    if a > 0.1:
                        a = 1
                    else:
                        a = -1
                    X[i, :] = pX[i, :] + 0.3 * np.abs(pX[i, :] - worse) + a * 0.1 * (XX[i, :])
                    X1[i, :] = pX1[i, :] + 0.3 * np.abs(pX1[i, :] - worse1) + a * 0.1 * (XX1[i, :])
                    X2[i, :] = pX2[i, :] + 0.3 * np.abs(pX2[i, :] - worse2) + a * 0.1 * (XX2[i, :])
                else:
                    aaa = np.random.randint(180, size=1)
                    if aaa == 0 or aaa == 90 or aaa == 180:
                        X[i, :] = pX[i, :]
                        X1[i, :] = pX1[i, :]
                        X2[i, :] = pX2[i, :]
                    theta = aaa * math.pi / 180
                    X[i, :] = pX[i, :] + math.tan(theta) * np.abs(pX[i, :] - XX[i, :])
                    X1[i, :] = pX1[i, :] + math.tan(theta) * np.abs(pX1[i, :] - XX1[i, :])
                    X2[i, :] = pX2[i, :] + math.tan(theta) * np.abs(pX2[i, :] - XX2[i, :])
                X[i, :] = Bounds(X[i, :], lb, ub)
                X1[i, :] = Bounds(X1[i, :], lb, ub)
                X2[i, :] = Bounds(X2[i, :], lb, ub)
                train_bags, train_labels = get_data(posi, nega, channel=2, window_size=301)
                model_type = "CNN"
                loss = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=2,
                                     window_size=301 + 6,
                                     model_file=model_type + '.301', batch_size=batch_size, n_epochs=n_epochs,
                                     num_filters=num_filters, learning_rate=X[i, 0], weight_decay=X[i, 1])
                fit[i, 0] = loss
                train_bags, train_labels = get_data(posi, nega, channel=2, window_size=301)
                model_type = "CNN-LSTM"
                loss1 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=2,
                                      window_size=301 + 6,
                                      model_file=model_type + '.301', batch_size=batch_size, n_epochs=n_epochs,
                                      num_filters=num_filters, learning_rate=X1[i, 0], weight_decay=X1[i, 1])
                fit1[i, 0] = loss1
                train_bags, train_labels = get_data(posi, nega, channel=2, window_size=301)
                model_type = "ResNet"
                loss2 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=2,
                                      window_size=301 + 6,
                                      model_file=model_type + '.301', batch_size=batch_size, n_epochs=n_epochs,
                                      num_filters=num_filters, learning_rate=X2[i, 0], weight_decay=X2[i, 1])
                fit2[i, 0] = loss2
            bestII = np.argmin(fit[:, 0])
            bestII1 = np.argmin(fit1[:, 0])
            bestII2 = np.argmin(fit2[:, 0])
            bestXX = X[bestII, :]
            bestXX1 = X1[bestII1, :]
            bestXX2 = X2[bestII2, :]
            R = 1 - t / M
            Xnew1 = bestXX * (1 - R)
            Xnew2 = bestXX * (1 + R)
            Xnew1l = bestXX1 * (1 - R)
            Xnew2l = bestXX1 * (1 + R)
            Xnew1r = bestXX2 * (1 - R)
            Xnew2r = bestXX2 * (1 + R)
            Xnew1 = Bounds(Xnew1, lb, ub)
            Xnew2 = Bounds(Xnew2, lb, ub)
            Xnew1l = Bounds(Xnew1l, lb, ub)
            Xnew2l = Bounds(Xnew2l, lb, ub)
            Xnew1r = Bounds(Xnew1r, lb, ub)
            Xnew2r = Bounds(Xnew2r, lb, ub)
            Xnew11 = bestX * (1 - R)
            Xnew22 = bestX * (1 + R)
            Xnew11l = bestX1 * (1 - R)
            Xnew22l = bestX1 * (1 + R)
            Xnew11r = bestX2 * (1 - R)
            Xnew22r = bestX2 * (1 + R)
            Xnew11 = Bounds(Xnew11, lb, ub)
            Xnew22 = Bounds(Xnew22, lb, ub)
            Xnew11l = Bounds(Xnew11l, lb, ub)
            Xnew22l = Bounds(Xnew22l, lb, ub)
            Xnew11r = Bounds(Xnew11r, lb, ub)
            Xnew22r = Bounds(Xnew22r, lb, ub)
            xLB = swapfun(Xnew1)
            xUB = swapfun(Xnew2)
            xLB1 = swapfun(Xnew1l)
            xUB1 = swapfun(Xnew2l)
            xLB2 = swapfun(Xnew1r)
            xUB2 = swapfun(Xnew2r)
            for i in range(pNum + 1, 12):
                X[i, :] = bestXX + (np.random.rand(1, dim)) * (pX[i, :] - Xnew1) + (np.random.rand(1, dim)) * (
                        pX[i, :] - Xnew2)
                X1[i, :] = bestXX1 + (np.random.rand(1, dim)) * (pX1[i, :] - Xnew1l) + (np.random.rand(1, dim)) * (
                        pX1[i, :] - Xnew2l)
                X2[i, :] = bestXX2 + (np.random.rand(1, dim)) * (pX2[i, :] - Xnew1r) + (np.random.rand(1, dim)) * (
                        pX2[i, :] - Xnew2r)
                X[i, :] = Bounds(X[i, :], xLB, xUB)
                X1[i, :] = Bounds(X1[i, :], xLB1, xUB1)
                X2[i, :] = Bounds(X2[i, :], xLB2, xUB2)
                train_bags, train_labels = get_data(posi, nega, channel=2, window_size=301)
                model_type = "CNN"
                loss = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=2,
                                     window_size=301 + 6,
                                     model_file=model_type + '.301', batch_size=batch_size, n_epochs=n_epochs,
                                     num_filters=num_filters, learning_rate=X[i, 0], weight_decay=X[i, 1])
                fit[i, 0] = loss
                train_bags, train_labels = get_data(posi, nega, channel=2, window_size=301)
                model_type = "CNN-LSTM"
                loss1 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=2,
                                      window_size=301 + 6,
                                      model_file=model_type + '.301', batch_size=batch_size, n_epochs=n_epochs,
                                      num_filters=num_filters, learning_rate=X1[i, 0], weight_decay=X1[i, 1])
                fit1[i, 0] = loss1
                train_bags, train_labels = get_data(posi, nega, channel=2, window_size=301)
                model_type = "ResNet"
                loss2 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=2,
                                      window_size=301 + 6,
                                      model_file=model_type + '.301', batch_size=batch_size, n_epochs=n_epochs,
                                      num_filters=num_filters, learning_rate=X2[i, 0], weight_decay=X2[i, 1])
                fit2[i, 0] = loss2
            for i in range(13, 19):
                X[i, :] = pX[i, :] + ((np.random.randn(1)) * (pX[i, :] - Xnew11) + (
                        (np.random.rand(1, dim)) * (pX[i, :] - Xnew22)))
                X1[i, :] = pX1[i, :] + ((np.random.randn(1)) * (pX1[i, :] - Xnew11l) + (
                        (np.random.rand(1, dim)) * (pX1[i, :] - Xnew22l)))
                X2[i, :] = pX2[i, :] + ((np.random.randn(1)) * (pX2[i, :] - Xnew11r) + (
                        (np.random.rand(1, dim)) * (pX2[i, :] - Xnew22r)))
                X[i, :] = Bounds(X[i, :], lb, ub)
                X1[i, :] = Bounds(X1[i, :], lb, ub)
                X2[i, :] = Bounds(X2[i, :], lb, ub)
                train_bags, train_labels = get_data(posi, nega, channel=2, window_size=301)
                model_type = "CNN"
                loss = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=2,
                                     window_size=301 + 6,
                                     model_file=model_type + '.301', batch_size=batch_size, n_epochs=n_epochs,
                                     num_filters=num_filters, learning_rate=X[i, 0], weight_decay=X[i, 1])
                fit[i, 0] = loss
                train_bags, train_labels = get_data(posi, nega, channel=2, window_size=301)
                model_type = "CNN-LSTM"
                loss1 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=2,
                                      window_size=301 + 6,
                                      model_file=model_type + '.301', batch_size=batch_size, n_epochs=n_epochs,
                                      num_filters=num_filters, learning_rate=X1[i, 0], weight_decay=X1[i, 1])
                fit1[i, 0] = loss1
                train_bags, train_labels = get_data(posi, nega, channel=2, window_size=301)
                model_type = "ResNet"
                loss2 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=2,
                                      window_size=301 + 6,
                                      model_file=model_type + '.301', batch_size=batch_size, n_epochs=n_epochs,
                                      num_filters=num_filters, learning_rate=X2[i, 0], weight_decay=X2[i, 1])
                fit2[i, 0] = loss2
            for j in range(20, pop):
                X[j, :] = levy(dim)*bestX + np.random.randn(1, dim) * (np.abs(pX[j, :] - bestXX) + np.abs(pX[j, :] - bestX)) / 2
                X1[j, :] = levy(dim)*bestX1 + np.random.randn(1, dim) * (
                        np.abs(pX1[j, :] - bestXX1) + np.abs(pX1[j, :] - bestX1)) / 2
                X2[j, :] = levy(dim)*bestX2 + np.random.randn(1, dim) * (
                        np.abs(pX2[j, :] - bestXX2) + np.abs(pX2[j, :] - bestX2)) / 2
                X[j, :] = Bounds(X[j, :], lb, ub)
                X1[j, :] = Bounds(X1[j, :], lb, ub)
                X2[j, :] = Bounds(X2[j, :], lb, ub)
                train_bags, train_labels = get_data(posi, nega, channel=2, window_size=301)
                model_type = "CNN"
                loss = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=2,
                                     window_size=301 + 6,
                                     model_file=model_type + '.301', batch_size=batch_size, n_epochs=n_epochs,
                                     num_filters=num_filters, learning_rate=X[j, 0], weight_decay=X[j, 1])
                fit[j, 0] = loss
                train_bags, train_labels = get_data(posi, nega, channel=2, window_size=301)
                model_type = "CNN-LSTM"
                loss1 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=2,
                                      window_size=301 + 6,
                                      model_file=model_type + '.301', batch_size=batch_size, n_epochs=n_epochs,
                                      num_filters=num_filters, learning_rate=X1[j, 0], weight_decay=X1[j, 1])
                fit1[j, 0] = loss1
                train_bags, train_labels = get_data(posi, nega, channel=2, window_size=301)
                model_type = "ResNet"
                loss2 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=2,
                                      window_size=301 + 6,
                                      model_file=model_type + '.301', batch_size=batch_size, n_epochs=n_epochs,
                                      num_filters=num_filters, learning_rate=X2[j, 0], weight_decay=X2[j, 1])
                fit2[j, 0] = loss2
            XX = pX
            XX1 = pX1
            XX2 = pX2
            for i in range(pop):
                if fit[i, 0] < pFit[i, 0]:
                    pFit[i, 0] = fit[i, 0]
                    pX[i, :] = X[i, :]
                if pFit[i, 0] < fMin:
                    fMin = pFit[i, 0]
                    bestX = pX[i, :]
            for i in range(pop):
                if fit1[i, 0] < pFit1[i, 0]:
                    pFit1[i, 0] = fit1[i, 0]
                    pX1[i, :] = X1[i, :]
                if pFit1[i, 0] < fMin1:
                    fMin1 = pFit1[i, 0]
                    bestX1 = pX1[i, :]
            for i in range(pop):
                if fit2[i, 0] < pFit2[i, 0]:
                    pFit2[i, 0] = fit2[i, 0]
                    pX2[i, :] = X2[i, :]
                if pFit2[i, 0] < fMin2:
                    fMin2 = pFit2[i, 0]
                    bestX2 = pX2[i, :]
            X = pX
            X1 = pX1
            X2 = pX2

        print("301")
        train_bags, train_labels = get_data(posi, nega, channel = 2, window_size = 301)
        model_type = "CNN"
        train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 2, window_size = 301 + 6, model_file = model_type + '.301', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters, learning_rate=bestX[0], weight_decay=bestX[1])
        model_type = "CNN-LSTM"
        train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 2, window_size = 301 + 6, model_file = model_type + '.301', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters, learning_rate=bestX1[0], weight_decay=bestX1[1])
        model_type = "ResNet"
        train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 2, window_size = 301 + 6, model_file = model_type + '.301', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters, learning_rate=bestX2[0], weight_decay=bestX2[1])

        pop = 30
        M = 1
        c = [0.0001, 1e-5]
        d = [5e-2, 1e-2]
        dim = 2
        P_percent = 0.2
        pNum = round(pop * P_percent)
        lb = c * np.ones((1, dim))
        ub = d * np.ones((1, dim))
        X = np.zeros((pop, dim))
        X1 = np.zeros((pop, dim))
        X2 = np.zeros((pop, dim))
        for i in range(pop):
            for j in range(dim):
                X[i, j] = X[i, j] + 1e-5 - np.mod(2.2 * np.sin(2 * np.pi * X[i, j]) / (2 * np.pi), 1)
                X1[i, j] = X1[i, j] + 1e-5 - np.mod(2.2 * np.sin(2 * np.pi * X1[i, j]) / (2 * np.pi), 1)
                X2[i, j] = X2[i, j] + 1e-5 - np.mod(2.2 * np.sin(2 * np.pi * X2[i, j]) / (2 * np.pi), 1)
                if j == 1:
                    if X[i, j] > 0.0001:
                        X[i, j] = 0.0001
                    if X1[i, j] > 0.0001:
                        X1[i, j] = 0.0001
                    if X2[i, j] > 0.0001:
                        X2[i, j] = 0.0001
                    if X[i, j] < 1e-5:
                        X[i, j] = 1e-5
                    if X1[i, j] < 1e-5:
                        X1[i, j] = 1e-5
                    if X2[i, j] < 1e-5:
                        X2[i, j] = 1e-5
                if j == 2:
                    if X[i, j] > 5e-2:
                        X[i, j] = 5e-2
                    if X1[i, j] > 5e-2:
                        X1[i, j] = 5e-2
                    if X2[i, j] > 5e-2:
                        X2[i, j] = 5e-2
                    if X[i, j] < 1e-2:
                        X[i, j] = 1e-2
                    if X1[i, j] < 1e-2:
                        X1[i, j] = 1e-2
                    if X2[i, j] < 1e-2:
                        X2[i, j] = 1e-2
        fit = np.zeros((pop, 1))
        fit1 = np.zeros((pop, 1))
        fit2 = np.zeros((pop, 1))

        for i in range(pop):
            X[i, :] = lb + (ub - lb) * np.random.rand(1, dim)
            train_bags, train_labels = get_data(posi, nega, channel=2, window_size=351)
            model_type = "CNN"
            loss = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=2,
                                 window_size=351 + 6,
                                 model_file=model_type + '.351', batch_size=batch_size, n_epochs=n_epochs,
                                 num_filters=num_filters, learning_rate=X[i, 0], weight_decay=X[i, 1])
            fit[i, 0] = loss

            train_bags, train_labels = get_data(posi, nega, channel=2, window_size=351)
            model_type = "CNN-LSTM"
            loss1 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=2,
                                  window_size=351 + 6,
                                  model_file=model_type + '.351', batch_size=batch_size, n_epochs=n_epochs,
                                  num_filters=num_filters, learning_rate=X1[i, 0], weight_decay=X1[i, 1])
            fit1[i, 0] = loss1

            train_bags, train_labels = get_data(posi, nega, channel=2, window_size=351)
            model_type = "ResNet"
            loss2 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=2,
                                  window_size=351 + 6,
                                  model_file=model_type + '.351', batch_size=batch_size, n_epochs=n_epochs,
                                  num_filters=num_filters, learning_rate=X2[i, 0], weight_decay=X2[i, 1])
            fit2[i, 0] = loss2
        pFit = fit
        pFit1 = fit1
        pFit2 = fit2
        pX = X
        pX1 = X1
        pX2 = X2
        XX = pX
        XX1 = pX1
        XX2 = pX2
        fMin = np.min(fit[:, 0])
        fMin1 = np.min(fit1[:, 0])
        fMin2 = np.min(fit2[:, 0])
        bestI = np.argmin(fit[:, 0])
        bestI1 = np.argmin(fit1[:, 0])
        bestI2 = np.argmin(fit2[:, 0])
        bestX = X[bestI, :]
        bestX1 = X1[bestI1, :]
        bestX2 = X2[bestI2, :]

        for t in range(M):
            B = np.argmax(pFit[:, 0])
            B1 = np.argmax(pFit1[:, 0])
            B2 = np.argmax(pFit2[:, 0])
            worse = X[B, :]
            worse1 = X1[B1, :]
            worse2 = X2[B2, :]
            r2 = np.random.rand(1)
            for i in range(pNum):
                if r2 < 0.9:
                    a = np.random.rand(1)
                    if a > 0.1:
                        a = 1
                    else:
                        a = -1
                    X[i, :] = pX[i, :] + 0.3 * np.abs(pX[i, :] - worse) + a * 0.1 * (XX[i, :])
                    X1[i, :] = pX1[i, :] + 0.3 * np.abs(pX1[i, :] - worse1) + a * 0.1 * (XX1[i, :])
                    X2[i, :] = pX2[i, :] + 0.3 * np.abs(pX2[i, :] - worse2) + a * 0.1 * (XX2[i, :])
                else:
                    aaa = np.random.randint(180, size=1)
                    if aaa == 0 or aaa == 90 or aaa == 180:
                        X[i, :] = pX[i, :]
                        X1[i, :] = pX1[i, :]
                        X2[i, :] = pX2[i, :]
                    theta = aaa * math.pi / 180
                    X[i, :] = pX[i, :] + math.tan(theta) * np.abs(pX[i, :] - XX[i, :])
                    X1[i, :] = pX1[i, :] + math.tan(theta) * np.abs(pX1[i, :] - XX1[i, :])
                    X2[i, :] = pX2[i, :] + math.tan(theta) * np.abs(pX2[i, :] - XX2[i, :])
                X[i, :] = Bounds(X[i, :], lb, ub)
                X1[i, :] = Bounds(X1[i, :], lb, ub)
                X2[i, :] = Bounds(X2[i, :], lb, ub)
                train_bags, train_labels = get_data(posi, nega, channel=2, window_size=351)
                model_type = "CNN"
                loss = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=2,
                                     window_size=351 + 6,
                                     model_file=model_type + '.351', batch_size=batch_size, n_epochs=n_epochs,
                                     num_filters=num_filters, learning_rate=X[i, 0], weight_decay=X[i, 1])
                fit[i, 0] = loss

                train_bags, train_labels = get_data(posi, nega, channel=2, window_size=351)
                model_type = "CNN-LSTM"
                loss1 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=2,
                                      window_size=351 + 6,
                                      model_file=model_type + '.351', batch_size=batch_size, n_epochs=n_epochs,
                                      num_filters=num_filters, learning_rate=X1[i, 0], weight_decay=X1[i, 1])
                fit1[i, 0] = loss1

                train_bags, train_labels = get_data(posi, nega, channel=2, window_size=351)
                model_type = "ResNet"
                loss2 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=2,
                                      window_size=351 + 6,
                                      model_file=model_type + '.351', batch_size=batch_size, n_epochs=n_epochs,
                                      num_filters=num_filters, learning_rate=X2[i, 0], weight_decay=X2[i, 1])
                fit2[i, 0] = loss2
            bestII = np.argmin(fit[:, 0])
            bestII1 = np.argmin(fit1[:, 0])
            bestII2 = np.argmin(fit2[:, 0])
            bestXX = X[bestII, :]
            bestXX1 = X1[bestII1, :]
            bestXX2 = X2[bestII2, :]
            R = 1 - t / M
            Xnew1 = bestXX * (1 - R)
            Xnew2 = bestXX * (1 + R)
            Xnew1l = bestXX1 * (1 - R)
            Xnew2l = bestXX1 * (1 + R)
            Xnew1r = bestXX2 * (1 - R)
            Xnew2r = bestXX2 * (1 + R)
            Xnew1 = Bounds(Xnew1, lb, ub)
            Xnew2 = Bounds(Xnew2, lb, ub)
            Xnew1l = Bounds(Xnew1l, lb, ub)
            Xnew2l = Bounds(Xnew2l, lb, ub)
            Xnew1r = Bounds(Xnew1r, lb, ub)
            Xnew2r = Bounds(Xnew2r, lb, ub)
            Xnew11 = bestX * (1 - R)
            Xnew22 = bestX * (1 + R)
            Xnew11l = bestX1 * (1 - R)
            Xnew22l = bestX1 * (1 + R)
            Xnew11r = bestX2 * (1 - R)
            Xnew22r = bestX2 * (1 + R)
            Xnew11 = Bounds(Xnew11, lb, ub)
            Xnew22 = Bounds(Xnew22, lb, ub)
            Xnew11l = Bounds(Xnew11l, lb, ub)
            Xnew22l = Bounds(Xnew22l, lb, ub)
            Xnew11r = Bounds(Xnew11r, lb, ub)
            Xnew22r = Bounds(Xnew22r, lb, ub)
            xLB = swapfun(Xnew1)
            xUB = swapfun(Xnew2)
            xLB1 = swapfun(Xnew1l)
            xUB1 = swapfun(Xnew2l)
            xLB2 = swapfun(Xnew1r)
            xUB2 = swapfun(Xnew2r)
            for i in range(pNum + 1, 12):
                X[i, :] = bestXX + (np.random.rand(1, dim)) * (pX[i, :] - Xnew1) + (np.random.rand(1, dim)) * (
                        pX[i, :] - Xnew2)
                X1[i, :] = bestXX1 + (np.random.rand(1, dim)) * (pX1[i, :] - Xnew1l) + (np.random.rand(1, dim)) * (
                        pX1[i, :] - Xnew2l)
                X2[i, :] = bestXX2 + (np.random.rand(1, dim)) * (pX2[i, :] - Xnew1r) + (np.random.rand(1, dim)) * (
                        pX2[i, :] - Xnew2r)
                X[i, :] = Bounds(X[i, :], xLB, xUB)
                X1[i, :] = Bounds(X1[i, :], xLB1, xUB1)
                X2[i, :] = Bounds(X2[i, :], xLB2, xUB2)
                train_bags, train_labels = get_data(posi, nega, channel=2, window_size=351)
                model_type = "CNN"
                loss = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=2,
                                     window_size=351 + 6,
                                     model_file=model_type + '.351', batch_size=batch_size, n_epochs=n_epochs,
                                     num_filters=num_filters, learning_rate=X[i, 0], weight_decay=X[i, 1])
                fit[i, 0] = loss

                train_bags, train_labels = get_data(posi, nega, channel=2, window_size=351)
                model_type = "CNN-LSTM"
                loss1 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=2,
                                      window_size=351 + 6,
                                      model_file=model_type + '.351', batch_size=batch_size, n_epochs=n_epochs,
                                      num_filters=num_filters, learning_rate=X1[i, 0], weight_decay=X1[i, 1])
                fit1[i, 0] = loss1

                train_bags, train_labels = get_data(posi, nega, channel=2, window_size=351)
                model_type = "ResNet"
                loss2 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=2,
                                      window_size=351 + 6,
                                      model_file=model_type + '.351', batch_size=batch_size, n_epochs=n_epochs,
                                      num_filters=num_filters, learning_rate=X2[i, 0], weight_decay=X2[i, 1])
                fit2[i, 0] = loss2
            for i in range(13, 19):
                X[i, :] = pX[i, :] + ((np.random.randn(1)) * (pX[i, :] - Xnew11) + (
                        (np.random.rand(1, dim)) * (pX[i, :] - Xnew22)))
                X1[i, :] = pX1[i, :] + ((np.random.randn(1)) * (pX1[i, :] - Xnew11l) + (
                        (np.random.rand(1, dim)) * (pX1[i, :] - Xnew22l)))
                X2[i, :] = pX2[i, :] + ((np.random.randn(1)) * (pX2[i, :] - Xnew11r) + (
                        (np.random.rand(1, dim)) * (pX2[i, :] - Xnew22r)))
                X[i, :] = Bounds(X[i, :], lb, ub)
                X1[i, :] = Bounds(X1[i, :], lb, ub)
                X2[i, :] = Bounds(X2[i, :], lb, ub)
                train_bags, train_labels = get_data(posi, nega, channel=2, window_size=351)
                model_type = "CNN"
                loss = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=2,
                                     window_size=351 + 6,
                                     model_file=model_type + '.351', batch_size=batch_size, n_epochs=n_epochs,
                                     num_filters=num_filters, learning_rate=X[i, 0], weight_decay=X[i, 1])
                fit[i, 0] = loss

                train_bags, train_labels = get_data(posi, nega, channel=2, window_size=351)
                model_type = "CNN-LSTM"
                loss1 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=2,
                                      window_size=351 + 6,
                                      model_file=model_type + '.351', batch_size=batch_size, n_epochs=n_epochs,
                                      num_filters=num_filters, learning_rate=X1[i, 0], weight_decay=X1[i, 1])
                fit1[i, 0] = loss1

                train_bags, train_labels = get_data(posi, nega, channel=2, window_size=351)
                model_type = "ResNet"
                loss2 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=2,
                                      window_size=351 + 6,
                                      model_file=model_type + '.351', batch_size=batch_size, n_epochs=n_epochs,
                                      num_filters=num_filters, learning_rate=X2[i, 0], weight_decay=X2[i, 1])
                fit2[i, 0] = loss2
            for j in range(20, pop):
                X[j, :] = levy(dim)*bestX + np.random.randn(1, dim) * (np.abs(pX[j, :] - bestXX) + np.abs(pX[j, :] - bestX)) / 2
                X1[j, :] = levy(dim)*bestX1 + np.random.randn(1, dim) * (
                        np.abs(pX1[j, :] - bestXX1) + np.abs(pX1[j, :] - bestX1)) / 2
                X2[j, :] = levy(dim)*bestX2 + np.random.randn(1, dim) * (
                        np.abs(pX2[j, :] - bestXX2) + np.abs(pX2[j, :] - bestX2)) / 2
                X[j, :] = Bounds(X[j, :], lb, ub)
                X1[j, :] = Bounds(X1[j, :], lb, ub)
                X2[j, :] = Bounds(X2[j, :], lb, ub)
                train_bags, train_labels = get_data(posi, nega, channel=2, window_size=351)
                model_type = "CNN"
                loss = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=2,
                                     window_size=351 + 6,
                                     model_file=model_type + '.351', batch_size=batch_size, n_epochs=n_epochs,
                                     num_filters=num_filters, learning_rate=X[j, 0], weight_decay=X[j, 1])
                fit[j, 0] = loss
                train_bags, train_labels = get_data(posi, nega, channel=2, window_size=351)
                model_type = "CNN-LSTM"
                loss1 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=2,
                                      window_size=351 + 6,
                                      model_file=model_type + '.351', batch_size=batch_size, n_epochs=n_epochs,
                                      num_filters=num_filters, learning_rate=X1[j, 0], weight_decay=X1[j, 1])
                fit1[j, 0] = loss1
                train_bags, train_labels = get_data(posi, nega, channel=2, window_size=351)
                model_type = "ResNet"
                loss2 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=2,
                                      window_size=351 + 6,
                                      model_file=model_type + '.351', batch_size=batch_size, n_epochs=n_epochs,
                                      num_filters=num_filters, learning_rate=X2[j, 0], weight_decay=X2[j, 1])
                fit2[j, 0] = loss2
            XX = pX
            XX1 = pX1
            XX2 = pX2
            for i in range(pop):
                if fit[i, 0] < pFit[i, 0]:
                    pFit[i, 0] = fit[i, 0]
                    pX[i, :] = X[i, :]
                if pFit[i, 0] < fMin:
                    fMin = pFit[i, 0]
                    bestX = pX[i, :]
            for i in range(pop):
                if fit1[i, 0] < pFit1[i, 0]:
                    pFit1[i, 0] = fit1[i, 0]
                    pX1[i, :] = X1[i, :]
                if pFit1[i, 0] < fMin1:
                    fMin1 = pFit1[i, 0]
                    bestX1 = pX1[i, :]
            for i in range(pop):
                if fit2[i, 0] < pFit2[i, 0]:
                    pFit2[i, 0] = fit2[i, 0]
                    pX2[i, :] = X2[i, :]
                if pFit2[i, 0] < fMin2:
                    fMin2 = pFit2[i, 0]
                    bestX2 = pX2[i, :]
            X = pX
            X1 = pX1
            X2 = pX2

        print("351")
        train_bags, train_labels = get_data(posi, nega, channel = 2, window_size = 351)
        model_type = "CNN"
        train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 2, window_size = 351 + 6, model_file = model_type + '.351', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters,learning_rate=bestX[0], weight_decay=bestX[1])
        model_type = "CNN-LSTM"
        train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 2, window_size = 351 + 6, model_file = model_type + '.351', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters, learning_rate=bestX1[0], weight_decay=bestX1[1])
        model_type = "ResNet"
        train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 2, window_size = 351 + 6, model_file = model_type + '.351', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters, learning_rate=bestX2[0], weight_decay=bestX2[1])

        pop = 30
        M = 1
        c = [0.0001, 1e-5]
        d = [5e-2, 1e-2]
        dim = 2
        P_percent = 0.2
        pNum = round(pop * P_percent)
        lb = c * np.ones((1, dim))
        ub = d * np.ones((1, dim))
        X = np.zeros((pop, dim))
        X1 = np.zeros((pop, dim))
        X2 = np.zeros((pop, dim))
        for i in range(pop):
            for j in range(dim):
                X[i, j] = X[i, j] + 1e-5 - np.mod(2.2 * np.sin(2 * np.pi * X[i, j]) / (2 * np.pi), 1)
                X1[i, j] = X1[i, j] + 1e-5 - np.mod(2.2 * np.sin(2 * np.pi * X1[i, j]) / (2 * np.pi), 1)
                X2[i, j] = X2[i, j] + 1e-5 - np.mod(2.2 * np.sin(2 * np.pi * X2[i, j]) / (2 * np.pi), 1)
                if j == 1:
                    if X[i, j] > 0.0001:
                        X[i, j] = 0.0001
                    if X1[i, j] > 0.0001:
                        X1[i, j] = 0.0001
                    if X2[i, j] > 0.0001:
                        X2[i, j] = 0.0001
                    if X[i, j] < 1e-5:
                        X[i, j] = 1e-5
                    if X1[i, j] < 1e-5:
                        X1[i, j] = 1e-5
                    if X2[i, j] < 1e-5:
                        X2[i, j] = 1e-5
                if j == 2:
                    if X[i, j] > 5e-2:
                        X[i, j] = 5e-2
                    if X1[i, j] > 5e-2:
                        X1[i, j] = 5e-2
                    if X2[i, j] > 5e-2:
                        X2[i, j] = 5e-2
                    if X[i, j] < 1e-2:
                        X[i, j] = 1e-2
                    if X1[i, j] < 1e-2:
                        X1[i, j] = 1e-2
                    if X2[i, j] < 1e-2:
                        X2[i, j] = 1e-2
        fit = np.zeros((pop, 1))
        fit1 = np.zeros((pop, 1))
        fit2 = np.zeros((pop, 1))

        for i in range(pop):
            train_bags, train_labels = get_data(posi, nega, channel=1, window_size=401)
            model_type = "CNN"
            loss = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=1,
                                 window_size=401 + 6,
                                 model_file=model_type + '.401', batch_size=batch_size, n_epochs=n_epochs,
                                 num_filters=num_filters, learning_rate=X[i, 0], weight_decay=X[i, 1])
            fit[i, 0] = loss
            train_bags, train_labels = get_data(posi, nega, channel=1, window_size=401)
            model_type = "CNN-LSTM"
            loss1 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=1,
                                  window_size=401 + 6,
                                  model_file=model_type + '.401', batch_size=batch_size, n_epochs=n_epochs,
                                  num_filters=num_filters, learning_rate=X1[i, 0], weight_decay=X1[i, 1])
            fit1[i, 0] = loss1
            train_bags, train_labels = get_data(posi, nega, channel=1, window_size=401)
            model_type = "ResNet"
            loss2 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=1,
                                  window_size=401 + 6,
                                  model_file=model_type + '.401', batch_size=batch_size, n_epochs=n_epochs,
                                  num_filters=num_filters, learning_rate=X2[i, 0], weight_decay=X2[i, 1])
            fit2[i, 0] = loss2
        pFit = fit
        pFit1 = fit1
        pFit2 = fit2
        pX = X
        pX1 = X1
        pX2 = X2
        XX = pX
        XX1 = pX1
        XX2 = pX2
        fMin = np.min(fit[:, 0])
        fMin1 = np.min(fit1[:, 0])
        fMin2 = np.min(fit2[:, 0])
        bestI = np.argmin(fit[:, 0])
        bestI1 = np.argmin(fit1[:, 0])
        bestI2 = np.argmin(fit2[:, 0])
        bestX = X[bestI, :]
        bestX1 = X1[bestI1, :]
        bestX2 = X2[bestI2, :]

        for t in range(M):
            B = np.argmax(pFit[:, 0])
            B1 = np.argmax(pFit1[:, 0])
            B2 = np.argmax(pFit2[:, 0])
            worse = X[B, :]
            worse1 = X1[B1, :]
            worse2 = X2[B2, :]
            r2 = np.random.rand(1)
            for i in range(pNum):
                if r2 < 0.9:
                    a = np.random.rand(1)
                    if a > 0.1:
                        a = 1
                    else:
                        a = -1
                    X[i, :] = pX[i, :] + 0.3 * np.abs(pX[i, :] - worse) + a * 0.1 * (XX[i, :])
                    X1[i, :] = pX1[i, :] + 0.3 * np.abs(pX1[i, :] - worse1) + a * 0.1 * (XX1[i, :])
                    X2[i, :] = pX2[i, :] + 0.3 * np.abs(pX2[i, :] - worse2) + a * 0.1 * (XX2[i, :])
                else:
                    aaa = np.random.randint(180, size=1)
                    if aaa == 0 or aaa == 90 or aaa == 180:
                        X[i, :] = pX[i, :]
                        X1[i, :] = pX1[i, :]
                        X2[i, :] = pX2[i, :]
                    theta = aaa * math.pi / 180
                    X[i, :] = pX[i, :] + math.tan(theta) * np.abs(pX[i, :] - XX[i, :])
                    X1[i, :] = pX1[i, :] + math.tan(theta) * np.abs(pX1[i, :] - XX1[i, :])
                    X2[i, :] = pX2[i, :] + math.tan(theta) * np.abs(pX2[i, :] - XX2[i, :])
                X[i, :] = Bounds(X[i, :], lb, ub)
                X1[i, :] = Bounds(X1[i, :], lb, ub)
                X2[i, :] = Bounds(X2[i, :], lb, ub)
                train_bags, train_labels = get_data(posi, nega, channel=1, window_size=401)
                model_type = "CNN"
                loss = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=1,
                                     window_size=401 + 6,
                                     model_file=model_type + '.401', batch_size=batch_size, n_epochs=n_epochs,
                                     num_filters=num_filters, learning_rate=X[i, 0], weight_decay=X[i, 1])
                fit[i, 0] = loss
                train_bags, train_labels = get_data(posi, nega, channel=1, window_size=401)
                model_type = "CNN-LSTM"
                loss1 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=1,
                                      window_size=401 + 6,
                                      model_file=model_type + '.401', batch_size=batch_size, n_epochs=n_epochs,
                                      num_filters=num_filters, learning_rate=X1[i, 0], weight_decay=X1[i, 1])
                fit1[i, 0] = loss1
                train_bags, train_labels = get_data(posi, nega, channel=1, window_size=401)
                model_type = "ResNet"
                loss2 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=1,
                                      window_size=401 + 6,
                                      model_file=model_type + '.401', batch_size=batch_size, n_epochs=n_epochs,
                                      num_filters=num_filters, learning_rate=X2[i, 0], weight_decay=X2[i, 1])
                fit2[i, 0] = loss2
            bestII = np.argmin(fit[:, 0])
            bestII1 = np.argmin(fit1[:, 0])
            bestII2 = np.argmin(fit2[:, 0])
            bestXX = X[bestII, :]
            bestXX1 = X1[bestII1, :]
            bestXX2 = X2[bestII2, :]
            R = 1 - t / M
            Xnew1 = bestXX * (1 - R)
            Xnew2 = bestXX * (1 + R)
            Xnew1l = bestXX1 * (1 - R)
            Xnew2l = bestXX1 * (1 + R)
            Xnew1r = bestXX2 * (1 - R)
            Xnew2r = bestXX2 * (1 + R)
            Xnew1 = Bounds(Xnew1, lb, ub)
            Xnew2 = Bounds(Xnew2, lb, ub)
            Xnew1l = Bounds(Xnew1l, lb, ub)
            Xnew2l = Bounds(Xnew2l, lb, ub)
            Xnew1r = Bounds(Xnew1r, lb, ub)
            Xnew2r = Bounds(Xnew2r, lb, ub)
            Xnew11 = bestX * (1 - R)
            Xnew22 = bestX * (1 + R)
            Xnew11l = bestX1 * (1 - R)
            Xnew22l = bestX1 * (1 + R)
            Xnew11r = bestX2 * (1 - R)
            Xnew22r = bestX2 * (1 + R)
            Xnew11 = Bounds(Xnew11, lb, ub)
            Xnew22 = Bounds(Xnew22, lb, ub)
            Xnew11l = Bounds(Xnew11l, lb, ub)
            Xnew22l = Bounds(Xnew22l, lb, ub)
            Xnew11r = Bounds(Xnew11r, lb, ub)
            Xnew22r = Bounds(Xnew22r, lb, ub)
            xLB = swapfun(Xnew1)
            xUB = swapfun(Xnew2)
            xLB1 = swapfun(Xnew1l)
            xUB1 = swapfun(Xnew2l)
            xLB2 = swapfun(Xnew1r)
            xUB2 = swapfun(Xnew2r)
            for i in range(pNum + 1, 12):
                X[i, :] = bestXX + (np.random.rand(1, dim)) * (pX[i, :] - Xnew1) + (np.random.rand(1, dim)) * (
                        pX[i, :] - Xnew2)
                X1[i, :] = bestXX1 + (np.random.rand(1, dim)) * (pX1[i, :] - Xnew1l) + (np.random.rand(1, dim)) * (
                        pX1[i, :] - Xnew2l)
                X2[i, :] = bestXX2 + (np.random.rand(1, dim)) * (pX2[i, :] - Xnew1r) + (np.random.rand(1, dim)) * (
                        pX2[i, :] - Xnew2r)
                X[i, :] = Bounds(X[i, :], xLB, xUB)
                X1[i, :] = Bounds(X1[i, :], xLB1, xUB1)
                X2[i, :] = Bounds(X2[i, :], xLB2, xUB2)
                train_bags, train_labels = get_data(posi, nega, channel=1, window_size=401)
                model_type = "CNN"
                loss = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=1,
                                     window_size=401 + 6,
                                     model_file=model_type + '.401', batch_size=batch_size, n_epochs=n_epochs,
                                     num_filters=num_filters, learning_rate=X[i, 0], weight_decay=X[i, 1])
                fit[i, 0] = loss
                train_bags, train_labels = get_data(posi, nega, channel=1, window_size=401)
                model_type = "CNN-LSTM"
                loss1 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=1,
                                      window_size=401 + 6,
                                      model_file=model_type + '.401', batch_size=batch_size, n_epochs=n_epochs,
                                      num_filters=num_filters, learning_rate=X1[i, 0], weight_decay=X1[i, 1])
                fit1[i, 0] = loss1
                train_bags, train_labels = get_data(posi, nega, channel=1, window_size=401)
                model_type = "ResNet"
                loss2 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=1,
                                      window_size=401 + 6,
                                      model_file=model_type + '.401', batch_size=batch_size, n_epochs=n_epochs,
                                      num_filters=num_filters, learning_rate=X2[i, 0], weight_decay=X2[i, 1])
                fit2[i, 0] = loss2
            for i in range(13, 19):
                X[i, :] = pX[i, :] + ((np.random.randn(1)) * (pX[i, :] - Xnew11) + (
                        (np.random.rand(1, dim)) * (pX[i, :] - Xnew22)))
                X1[i, :] = pX1[i, :] + ((np.random.randn(1)) * (pX1[i, :] - Xnew11l) + (
                        (np.random.rand(1, dim)) * (pX1[i, :] - Xnew22l)))
                X2[i, :] = pX2[i, :] + ((np.random.randn(1)) * (pX2[i, :] - Xnew11r) + (
                        (np.random.rand(1, dim)) * (pX2[i, :] - Xnew22r)))
                X[i, :] = Bounds(X[i, :], lb, ub)
                X1[i, :] = Bounds(X1[i, :], lb, ub)
                X2[i, :] = Bounds(X2[i, :], lb, ub)
                train_bags, train_labels = get_data(posi, nega, channel=1, window_size=401)
                model_type = "CNN"
                loss = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=1,
                                     window_size=401 + 6,
                                     model_file=model_type + '.401', batch_size=batch_size, n_epochs=n_epochs,
                                     num_filters=num_filters, learning_rate=X[i, 0], weight_decay=X[i, 1])
                fit[i, 0] = loss
                train_bags, train_labels = get_data(posi, nega, channel=1, window_size=401)
                model_type = "CNN-LSTM"
                loss1 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=1,
                                      window_size=401 + 6,
                                      model_file=model_type + '.401', batch_size=batch_size, n_epochs=n_epochs,
                                      num_filters=num_filters, learning_rate=X1[i, 0], weight_decay=X1[i, 1])
                fit1[i, 0] = loss1
                train_bags, train_labels = get_data(posi, nega, channel=1, window_size=401)
                model_type = "ResNet"
                loss2 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=1,
                                      window_size=401 + 6,
                                      model_file=model_type + '.401', batch_size=batch_size, n_epochs=n_epochs,
                                      num_filters=num_filters, learning_rate=X2[i, 0], weight_decay=X2[i, 1])
                fit2[i, 0] = loss2
            for j in range(20, pop):
                X[j, :] = levy(dim)*bestX + np.random.randn(1, dim) * (np.abs(pX[j, :] - bestXX) + np.abs(pX[j, :] - bestX)) / 2
                X1[j, :] = levy(dim)*bestX1 + np.random.randn(1, dim) * (
                        np.abs(pX1[j, :] - bestXX1) + np.abs(pX1[j, :] - bestX1)) / 2
                X2[j, :] = levy(dim)*bestX2 + np.random.randn(1, dim) * (
                        np.abs(pX2[j, :] - bestXX2) + np.abs(pX2[j, :] - bestX2)) / 2
                X[j, :] = Bounds(X[j, :], lb, ub)
                X1[j, :] = Bounds(X1[j, :], lb, ub)
                X2[j, :] = Bounds(X2[j, :], lb, ub)
                train_bags, train_labels = get_data(posi, nega, channel=1, window_size=401)
                model_type = "CNN"
                loss = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=1,
                                     window_size=401 + 6,
                                     model_file=model_type + '.401', batch_size=batch_size, n_epochs=n_epochs,
                                     num_filters=num_filters, learning_rate=X[j, 0], weight_decay=X[j, 1])
                fit[j, 0] = loss
                train_bags, train_labels = get_data(posi, nega, channel=1, window_size=401)
                model_type = "CNN-LSTM"
                loss1 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=1,
                                      window_size=401 + 6,
                                      model_file=model_type + '.401', batch_size=batch_size, n_epochs=n_epochs,
                                      num_filters=num_filters, learning_rate=X1[j, 0], weight_decay=X1[j, 1])
                fit1[j, 0] = loss1
                train_bags, train_labels = get_data(posi, nega, channel=1, window_size=401)
                model_type = "ResNet"
                loss2 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=1,
                                      window_size=401 + 6,
                                      model_file=model_type + '.401', batch_size=batch_size, n_epochs=n_epochs,
                                      num_filters=num_filters, learning_rate=X2[j, 0], weight_decay=X2[j, 1])
                fit2[j, 0] = loss2
            XX = pX
            XX1 = pX1
            XX2 = pX2
            for i in range(pop):
                if fit[i, 0] < pFit[i, 0]:
                    pFit[i, 0] = fit[i, 0]
                    pX[i, :] = X[i, :]
                if pFit[i, 0] < fMin:
                    fMin = pFit[i, 0]
                    bestX = pX[i, :]
            for i in range(pop):
                if fit1[i, 0] < pFit1[i, 0]:
                    pFit1[i, 0] = fit1[i, 0]
                    pX1[i, :] = X1[i, :]
                if pFit1[i, 0] < fMin1:
                    fMin1 = pFit1[i, 0]
                    bestX1 = pX1[i, :]
            for i in range(pop):
                if fit2[i, 0] < pFit2[i, 0]:
                    pFit2[i, 0] = fit2[i, 0]
                    pX2[i, :] = X2[i, :]
                if pFit2[i, 0] < fMin2:
                    fMin2 = pFit2[i, 0]
                    bestX2 = pX2[i, :]
            X = pX
            X1 = pX1
            X2 = pX2

        print("401")
        train_bags, train_labels = get_data(posi, nega, channel = 1, window_size = 401)
        model_type = "CNN"
        train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 1, window_size = 401 + 6, model_file = model_type + '.401', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters, learning_rate=bestX[0], weight_decay=bestX[1])
        model_type = "CNN-LSTM"
        train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 1, window_size = 401 + 6, model_file = model_type + '.401', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters, learning_rate=bestX1[0], weight_decay=bestX1[1])
        model_type = "ResNet"
        train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 1, window_size = 401 + 6, model_file = model_type + '.401', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters, learning_rate=bestX2[0], weight_decay=bestX2[1])

        pop = 30
        M = 1
        c = [0.0001, 1e-5]
        d = [5e-2, 1e-2]
        dim = 2
        P_percent = 0.2
        pNum = round(pop * P_percent)
        lb = c * np.ones((1, dim))
        ub = d * np.ones((1, dim))
        X = np.zeros((pop, dim))
        X1 = np.zeros((pop, dim))
        X2 = np.zeros((pop, dim))
        for i in range(pop):
            for j in range(dim):
                X[i, j] = X[i, j] + 1e-5 - np.mod(2.2 * np.sin(2 * np.pi * X[i, j]) / (2 * np.pi), 1)
                X1[i, j] = X1[i, j] + 1e-5 - np.mod(2.2 * np.sin(2 * np.pi * X1[i, j]) / (2 * np.pi), 1)
                X2[i, j] = X2[i, j] + 1e-5 - np.mod(2.2 * np.sin(2 * np.pi * X2[i, j]) / (2 * np.pi), 1)
                if j == 1:
                    if X[i, j] > 0.0001:
                        X[i, j] = 0.0001
                    if X1[i, j] > 0.0001:
                        X1[i, j] = 0.0001
                    if X2[i, j] > 0.0001:
                        X2[i, j] = 0.0001
                    if X[i, j] < 1e-5:
                        X[i, j] = 1e-5
                    if X1[i, j] < 1e-5:
                        X1[i, j] = 1e-5
                    if X2[i, j] < 1e-5:
                        X2[i, j] = 1e-5
                if j == 2:
                    if X[i, j] > 5e-2:
                        X[i, j] = 5e-2
                    if X1[i, j] > 5e-2:
                        X1[i, j] = 5e-2
                    if X2[i, j] > 5e-2:
                        X2[i, j] = 5e-2
                    if X[i, j] < 1e-2:
                        X[i, j] = 1e-2
                    if X1[i, j] < 1e-2:
                        X1[i, j] = 1e-2
                    if X2[i, j] < 1e-2:
                        X2[i, j] = 1e-2
        fit = np.zeros((pop, 1))
        fit1 = np.zeros((pop, 1))
        fit2 = np.zeros((pop, 1))

        for i in range(pop):
            train_bags, train_labels = get_data(posi, nega, channel=1, window_size=451)
            model_type = "CNN"
            loss = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=1,
                                 window_size=451 + 6,
                                 model_file=model_type + '.451', batch_size=batch_size, n_epochs=n_epochs,
                                 num_filters=num_filters, learning_rate=X[i, 0], weight_decay=X[i, 1])
            fit[i, 0] = loss
            train_bags, train_labels = get_data(posi, nega, channel=1, window_size=451)
            model_type = "CNN-LSTM"
            loss1 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=1,
                                  window_size=451 + 6,
                                  model_file=model_type + '.451', batch_size=batch_size, n_epochs=n_epochs,
                                  num_filters=num_filters, learning_rate=X1[i, 0], weight_decay=X1[i, 1])
            fit1[i, 0] = loss1
            train_bags, train_labels = get_data(posi, nega, channel=1, window_size=451)
            model_type = "ResNet"
            loss2 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=1,
                                  window_size=451 + 6,
                                  model_file=model_type + '.451', batch_size=batch_size, n_epochs=n_epochs,
                                  num_filters=num_filters, learning_rate=X2[i, 0], weight_decay=X2[i, 1])
            fit2[i, 0] = loss2
        pFit = fit
        pFit1 = fit1
        pFit2 = fit2
        pX = X
        pX1 = X1
        pX2 = X2
        XX = pX
        XX1 = pX1
        XX2 = pX2
        fMin = np.min(fit[:, 0])
        fMin1 = np.min(fit1[:, 0])
        fMin2 = np.min(fit2[:, 0])
        bestI = np.argmin(fit[:, 0])
        bestI1 = np.argmin(fit1[:, 0])
        bestI2 = np.argmin(fit2[:, 0])
        bestX = X[bestI, :]
        bestX1 = X1[bestI1, :]
        bestX2 = X2[bestI2, :]

        for t in range(M):
            B = np.argmax(pFit[:, 0])
            B1 = np.argmax(pFit1[:, 0])
            B2 = np.argmax(pFit2[:, 0])
            worse = X[B, :]
            worse1 = X1[B1, :]
            worse2 = X2[B2, :]
            r2 = np.random.rand(1)
            for i in range(pNum):
                if r2 < 0.9:
                    a = np.random.rand(1)
                    if a > 0.1:
                        a = 1
                    else:
                        a = -1
                    X[i, :] = pX[i, :] + 0.3 * np.abs(pX[i, :] - worse) + a * 0.1 * (XX[i, :])
                    X1[i, :] = pX1[i, :] + 0.3 * np.abs(pX1[i, :] - worse1) + a * 0.1 * (XX1[i, :])
                    X2[i, :] = pX2[i, :] + 0.3 * np.abs(pX2[i, :] - worse2) + a * 0.1 * (XX2[i, :])
                else:
                    aaa = np.random.randint(180, size=1)
                    if aaa == 0 or aaa == 90 or aaa == 180:
                        X[i, :] = pX[i, :]
                        X1[i, :] = pX1[i, :]
                        X2[i, :] = pX2[i, :]
                    theta = aaa * math.pi / 180
                    X[i, :] = pX[i, :] + math.tan(theta) * np.abs(pX[i, :] - XX[i, :])
                    X1[i, :] = pX1[i, :] + math.tan(theta) * np.abs(pX1[i, :] - XX1[i, :])
                    X2[i, :] = pX2[i, :] + math.tan(theta) * np.abs(pX2[i, :] - XX2[i, :])
                X[i, :] = Bounds(X[i, :], lb, ub)
                X1[i, :] = Bounds(X1[i, :], lb, ub)
                X2[i, :] = Bounds(X2[i, :], lb, ub)
                train_bags, train_labels = get_data(posi, nega, channel=1, window_size=401)
                model_type = "CNN"
                loss = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=1,
                                     window_size=401 + 6,
                                     model_file=model_type + '.401', batch_size=batch_size, n_epochs=n_epochs,
                                     num_filters=num_filters, learning_rate=X[i, 0], weight_decay=X[i, 1])
                fit[i, 0] = loss
                train_bags, train_labels = get_data(posi, nega, channel=1, window_size=451)
                model_type = "CNN"
                loss = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=1,
                                     window_size=451 + 6,
                                     model_file=model_type + '.451', batch_size=batch_size, n_epochs=n_epochs,
                                     num_filters=num_filters, learning_rate=X[i, 0], weight_decay=X[i, 1])
                fit[i, 0] = loss
                train_bags, train_labels = get_data(posi, nega, channel=1, window_size=451)
                model_type = "CNN-LSTM"
                loss1 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=1,
                                      window_size=451 + 6,
                                      model_file=model_type + '.451', batch_size=batch_size, n_epochs=n_epochs,
                                      num_filters=num_filters, learning_rate=X1[i, 0], weight_decay=X1[i, 1])
                fit1[i, 0] = loss1
                train_bags, train_labels = get_data(posi, nega, channel=1, window_size=451)
                model_type = "ResNet"
                loss2 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=1,
                                      window_size=451 + 6,
                                      model_file=model_type + '.451', batch_size=batch_size, n_epochs=n_epochs,
                                      num_filters=num_filters, learning_rate=X2[i, 0], weight_decay=X2[i, 1])
                fit2[i, 0] = loss2
            bestII = np.argmin(fit[:, 0])
            bestII1 = np.argmin(fit1[:, 0])
            bestII2 = np.argmin(fit2[:, 0])
            bestXX = X[bestII, :]
            bestXX1 = X1[bestII1, :]
            bestXX2 = X2[bestII2, :]
            R = 1 - t / M
            Xnew1 = bestXX * (1 - R)
            Xnew2 = bestXX * (1 + R)
            Xnew1l = bestXX1 * (1 - R)
            Xnew2l = bestXX1 * (1 + R)
            Xnew1r = bestXX2 * (1 - R)
            Xnew2r = bestXX2 * (1 + R)
            Xnew1 = Bounds(Xnew1, lb, ub)
            Xnew2 = Bounds(Xnew2, lb, ub)
            Xnew1l = Bounds(Xnew1l, lb, ub)
            Xnew2l = Bounds(Xnew2l, lb, ub)
            Xnew1r = Bounds(Xnew1r, lb, ub)
            Xnew2r = Bounds(Xnew2r, lb, ub)
            Xnew11 = bestX * (1 - R)
            Xnew22 = bestX * (1 + R)
            Xnew11l = bestX1 * (1 - R)
            Xnew22l = bestX1 * (1 + R)
            Xnew11r = bestX2 * (1 - R)
            Xnew22r = bestX2 * (1 + R)
            Xnew11 = Bounds(Xnew11, lb, ub)
            Xnew22 = Bounds(Xnew22, lb, ub)
            Xnew11l = Bounds(Xnew11l, lb, ub)
            Xnew22l = Bounds(Xnew22l, lb, ub)
            Xnew11r = Bounds(Xnew11r, lb, ub)
            Xnew22r = Bounds(Xnew22r, lb, ub)
            xLB = swapfun(Xnew1)
            xUB = swapfun(Xnew2)
            xLB1 = swapfun(Xnew1l)
            xUB1 = swapfun(Xnew2l)
            xLB2 = swapfun(Xnew1r)
            xUB2 = swapfun(Xnew2r)
            for i in range(pNum + 1, 12):
                X[i, :] = bestXX + (np.random.rand(1, dim)) * (pX[i, :] - Xnew1) + (np.random.rand(1, dim)) * (
                        pX[i, :] - Xnew2)
                X1[i, :] = bestXX1 + (np.random.rand(1, dim)) * (pX1[i, :] - Xnew1l) + (np.random.rand(1, dim)) * (
                        pX1[i, :] - Xnew2l)
                X2[i, :] = bestXX2 + (np.random.rand(1, dim)) * (pX2[i, :] - Xnew1r) + (np.random.rand(1, dim)) * (
                        pX2[i, :] - Xnew2r)
                X[i, :] = Bounds(X[i, :], xLB, xUB)
                X1[i, :] = Bounds(X1[i, :], xLB1, xUB1)
                X2[i, :] = Bounds(X2[i, :], xLB2, xUB2)
                train_bags, train_labels = get_data(posi, nega, channel=1, window_size=451)
                model_type = "CNN"
                loss = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=1,
                                     window_size=451 + 6,
                                     model_file=model_type + '.451', batch_size=batch_size, n_epochs=n_epochs,
                                     num_filters=num_filters, learning_rate=X[i, 0], weight_decay=X[i, 1])
                fit[i, 0] = loss
                train_bags, train_labels = get_data(posi, nega, channel=1, window_size=451)
                model_type = "CNN-LSTM"
                loss1 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=1,
                                      window_size=451 + 6,
                                      model_file=model_type + '.451', batch_size=batch_size, n_epochs=n_epochs,
                                      num_filters=num_filters, learning_rate=X1[i, 0], weight_decay=X1[i, 1])
                fit1[i, 0] = loss1
                train_bags, train_labels = get_data(posi, nega, channel=1, window_size=451)
                model_type = "ResNet"
                loss2 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=1,
                                      window_size=451 + 6,
                                      model_file=model_type + '.451', batch_size=batch_size, n_epochs=n_epochs,
                                      num_filters=num_filters, learning_rate=X2[i, 0], weight_decay=X2[i, 1])
                fit2[i, 0] = loss2
            for i in range(13, 19):
                X[i, :] = pX[i, :] + ((np.random.randn(1)) * (pX[i, :] - Xnew11) + (
                        (np.random.rand(1, dim)) * (pX[i, :] - Xnew22)))
                X1[i, :] = pX1[i, :] + ((np.random.randn(1)) * (pX1[i, :] - Xnew11l) + (
                        (np.random.rand(1, dim)) * (pX1[i, :] - Xnew22l)))
                X2[i, :] = pX2[i, :] + ((np.random.randn(1)) * (pX2[i, :] - Xnew11r) + (
                        (np.random.rand(1, dim)) * (pX2[i, :] - Xnew22r)))
                X[i, :] = Bounds(X[i, :], lb, ub)
                X1[i, :] = Bounds(X1[i, :], lb, ub)
                X2[i, :] = Bounds(X2[i, :], lb, ub)
                train_bags, train_labels = get_data(posi, nega, channel=1, window_size=451)
                model_type = "CNN"
                loss = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=1,
                                     window_size=451 + 6,
                                     model_file=model_type + '.451', batch_size=batch_size, n_epochs=n_epochs,
                                     num_filters=num_filters, learning_rate=X[i, 0], weight_decay=X[i, 1])
                fit[i, 0] = loss
                train_bags, train_labels = get_data(posi, nega, channel=1, window_size=451)
                model_type = "CNN-LSTM"
                loss1 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=1,
                                      window_size=451 + 6,
                                      model_file=model_type + '.451', batch_size=batch_size, n_epochs=n_epochs,
                                      num_filters=num_filters, learning_rate=X1[i, 0], weight_decay=X1[i, 1])
                fit1[i, 0] = loss1
                train_bags, train_labels = get_data(posi, nega, channel=1, window_size=451)
                model_type = "ResNet"
                loss2 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=1,
                                      window_size=451 + 6,
                                      model_file=model_type + '.451', batch_size=batch_size, n_epochs=n_epochs,
                                      num_filters=num_filters, learning_rate=X2[i, 0], weight_decay=X2[i, 1])
                fit2[i, 0] = loss2
            for j in range(20, pop):
                X[j, :] = levy(dim)*bestX + np.random.randn(1, dim) * (np.abs(pX[j, :] - bestXX) + np.abs(pX[j, :] - bestX)) / 2
                X1[j, :] = levy(dim)*bestX1 + np.random.randn(1, dim) * (
                        np.abs(pX1[j, :] - bestXX1) + np.abs(pX1[j, :] - bestX1)) / 2
                X2[j, :] = levy(dim)*bestX2 + np.random.randn(1, dim) * (
                        np.abs(pX2[j, :] - bestXX2) + np.abs(pX2[j, :] - bestX2)) / 2
                X[j, :] = Bounds(X[j, :], lb, ub)
                X1[j, :] = Bounds(X1[j, :], lb, ub)
                X2[j, :] = Bounds(X2[j, :], lb, ub)
                train_bags, train_labels = get_data(posi, nega, channel=1, window_size=451)
                model_type = "CNN"
                loss = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=1,
                                     window_size=451 + 6,
                                     model_file=model_type + '.451', batch_size=batch_size, n_epochs=n_epochs,
                                     num_filters=num_filters, learning_rate=X[j, 0], weight_decay=X[j, 1])
                fit[j, 0] = loss
                train_bags, train_labels = get_data(posi, nega, channel=1, window_size=451)
                model_type = "CNN-LSTM"
                loss1 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=1,
                                      window_size=451 + 6,
                                      model_file=model_type + '.451', batch_size=batch_size, n_epochs=n_epochs,
                                      num_filters=num_filters, learning_rate=X1[j, 0], weight_decay=X1[j, 1])
                fit1[j, 0] = loss1
                train_bags, train_labels = get_data(posi, nega, channel=1, window_size=451)
                model_type = "ResNet"
                loss2 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=1,
                                      window_size=451 + 6,
                                      model_file=model_type + '.451', batch_size=batch_size, n_epochs=n_epochs,
                                      num_filters=num_filters, learning_rate=X2[j, 0], weight_decay=X2[j, 1])
                fit2[j, 0] = loss2
            XX = pX
            XX1 = pX1
            XX2 = pX2
            for i in range(pop):
                if fit[i, 0] < pFit[i, 0]:
                    pFit[i, 0] = fit[i, 0]
                    pX[i, :] = X[i, :]
                if pFit[i, 0] < fMin:
                    fMin = pFit[i, 0]
                    bestX = pX[i, :]
            for i in range(pop):
                if fit1[i, 0] < pFit1[i, 0]:
                    pFit1[i, 0] = fit1[i, 0]
                    pX1[i, :] = X1[i, :]
                if pFit1[i, 0] < fMin1:
                    fMin1 = pFit1[i, 0]
                    bestX1 = pX1[i, :]
            for i in range(pop):
                if fit2[i, 0] < pFit2[i, 0]:
                    pFit2[i, 0] = fit2[i, 0]
                    pX2[i, :] = X2[i, :]
                if pFit2[i, 0] < fMin2:
                    fMin2 = pFit2[i, 0]
                    bestX2 = pX2[i, :]
            X = pX
            X1 = pX1
            X2 = pX2

        print("451")
        train_bags, train_labels = get_data(posi, nega, channel = 1, window_size = 451)
        model_type = "CNN"
        train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 1, window_size = 451 + 6, model_file = model_type + '.451', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters, learning_rate=bestX[0], weight_decay=bestX[1])
        model_type = "CNN-LSTM"
        train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 1, window_size = 451 + 6, model_file = model_type + '.451', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters,learning_rate=bestX1[0], weight_decay=bestX1[1])
        model_type = "ResNet"
        train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 1, window_size = 451 + 6, model_file = model_type + '.451', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters, learning_rate=bestX2[0], weight_decay=bestX2[1])

        pop = 30
        M = 1
        c = [0.0001, 1e-5]
        d = [5e-2, 1e-2]
        dim = 2
        P_percent = 0.2
        pNum = round(pop * P_percent)
        lb = c * np.ones((1, dim))
        ub = d * np.ones((1, dim))
        X = np.zeros((pop, dim))
        X1 = np.zeros((pop, dim))
        X2 = np.zeros((pop, dim))
        for i in range(pop):
            for j in range(dim):
                X[i, j] = X[i, j] + 1e-5 - np.mod(2.2 * np.sin(2 * np.pi * X[i, j]) / (2 * np.pi), 1)
                X1[i, j] = X1[i, j] + 1e-5 - np.mod(2.2 * np.sin(2 * np.pi * X1[i, j]) / (2 * np.pi), 1)
                X2[i, j] = X2[i, j] + 1e-5 - np.mod(2.2 * np.sin(2 * np.pi * X2[i, j]) / (2 * np.pi), 1)
                if j == 1:
                    if X[i, j] > 0.0001:
                        X[i, j] = 0.0001
                    if X1[i, j] > 0.0001:
                        X1[i, j] = 0.0001
                    if X2[i, j] > 0.0001:
                        X2[i, j] = 0.0001
                    if X[i, j] < 1e-5:
                        X[i, j] = 1e-5
                    if X1[i, j] < 1e-5:
                        X1[i, j] = 1e-5
                    if X2[i, j] < 1e-5:
                        X2[i, j] = 1e-5
                if j == 2:
                    if X[i, j] > 5e-2:
                        X[i, j] = 5e-2
                    if X1[i, j] > 5e-2:
                        X1[i, j] = 5e-2
                    if X2[i, j] > 5e-2:
                        X2[i, j] = 5e-2
                    if X[i, j] < 1e-2:
                        X[i, j] = 1e-2
                    if X1[i, j] < 1e-2:
                        X1[i, j] = 1e-2
                    if X2[i, j] < 1e-2:
                        X2[i, j] = 1e-2
        fit = np.zeros((pop, 1))
        fit1 = np.zeros((pop, 1))
        fit2 = np.zeros((pop, 1))

        for i in range(pop):
            X[i, :] = lb + (ub - lb) * np.random.rand(1, dim)
            train_bags, train_labels = get_data(posi, nega, channel=1, window_size=501)
            model_type = "CNN"
            loss = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=1,
                                 window_size=501 + 6,
                                 model_file=model_type + '.501', batch_size=batch_size, n_epochs=n_epochs,
                                 num_filters=num_filters, learning_rate=X[i, 0], weight_decay=X[i, 1])
            fit[i, 0] = loss
            train_bags, train_labels = get_data(posi, nega, channel=1, window_size=501)
            model_type = "CNN-LSTM"
            loss1 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=1,
                                  window_size=501 + 6,
                                  model_file=model_type + '.501', batch_size=batch_size, n_epochs=n_epochs,
                                  num_filters=num_filters, learning_rate=X1[i, 0], weight_decay=X1[i, 1])
            fit1[i, 0] = loss1
            train_bags, train_labels = get_data(posi, nega, channel=1, window_size=501)
            model_type = "ResNet"
            loss2 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=1,
                                  window_size=501 + 6,
                                  model_file=model_type + '.501', batch_size=batch_size, n_epochs=n_epochs,
                                  num_filters=num_filters, learning_rate=X2[i, 0], weight_decay=X2[i, 1])
            fit2[i, 0] = loss2
        pFit = fit
        pFit1 = fit1
        pFit2 = fit2
        pX = X
        pX1 = X1
        pX2 = X2
        XX = pX
        XX1 = pX1
        XX2 = pX2
        fMin = np.min(fit[:, 0])
        fMin1 = np.min(fit1[:, 0])
        fMin2 = np.min(fit2[:, 0])
        bestI = np.argmin(fit[:, 0])
        bestI1 = np.argmin(fit1[:, 0])
        bestI2 = np.argmin(fit2[:, 0])
        bestX = X[bestI, :]
        bestX1 = X1[bestI1, :]
        bestX2 = X2[bestI2, :]

        for t in range(M):
            B = np.argmax(pFit[:, 0])
            B1 = np.argmax(pFit1[:, 0])
            B2 = np.argmax(pFit2[:, 0])
            worse = X[B, :]
            worse1 = X1[B1, :]
            worse2 = X2[B2, :]
            r2 = np.random.rand(1)
            for i in range(pNum):
                if r2 < 0.9:
                    a = np.random.rand(1)
                    if a > 0.1:
                        a = 1
                    else:
                        a = -1
                    X[i, :] = pX[i, :] + 0.3 * np.abs(pX[i, :] - worse) + a * 0.1 * (XX[i, :])
                    X1[i, :] = pX1[i, :] + 0.3 * np.abs(pX1[i, :] - worse1) + a * 0.1 * (XX1[i, :])
                    X2[i, :] = pX2[i, :] + 0.3 * np.abs(pX2[i, :] - worse2) + a * 0.1 * (XX2[i, :])
                else:
                    aaa = np.random.randint(180, size=1)
                    if aaa == 0 or aaa == 90 or aaa == 180:
                        X[i, :] = pX[i, :]
                        X1[i, :] = pX1[i, :]
                        X2[i, :] = pX2[i, :]
                    theta = aaa * math.pi / 180
                    X[i, :] = pX[i, :] + math.tan(theta) * np.abs(pX[i, :] - XX[i, :])
                    X1[i, :] = pX1[i, :] + math.tan(theta) * np.abs(pX1[i, :] - XX1[i, :])
                    X2[i, :] = pX2[i, :] + math.tan(theta) * np.abs(pX2[i, :] - XX2[i, :])
                X[i, :] = Bounds(X[i, :], lb, ub)
                X1[i, :] = Bounds(X1[i, :], lb, ub)
                X2[i, :] = Bounds(X2[i, :], lb, ub)
                train_bags, train_labels = get_data(posi, nega, channel=1, window_size=501)
                model_type = "CNN"
                loss = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=1,
                                     window_size=501 + 6,
                                     model_file=model_type + '.501', batch_size=batch_size, n_epochs=n_epochs,
                                     num_filters=num_filters, learning_rate=X[i, 0], weight_decay=X[i, 1])
                fit[i, 0] = loss
                train_bags, train_labels = get_data(posi, nega, channel=1, window_size=501)
                model_type = "CNN-LSTM"
                loss1 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=1,
                                      window_size=501 + 6,
                                      model_file=model_type + '.501', batch_size=batch_size, n_epochs=n_epochs,
                                      num_filters=num_filters, learning_rate=X1[i, 0], weight_decay=X1[i, 1])
                fit1[i, 0] = loss1
                train_bags, train_labels = get_data(posi, nega, channel=1, window_size=501)
                model_type = "ResNet"
                loss2 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=1,
                                      window_size=501 + 6,
                                      model_file=model_type + '.501', batch_size=batch_size, n_epochs=n_epochs,
                                      num_filters=num_filters, learning_rate=X2[i, 0], weight_decay=X2[i, 1])
                fit2[i, 0] = loss2
            bestII = np.argmin(fit[:, 0])
            bestII1 = np.argmin(fit1[:, 0])
            bestII2 = np.argmin(fit2[:, 0])
            bestXX = X[bestII, :]
            bestXX1 = X1[bestII1, :]
            bestXX2 = X2[bestII2, :]
            R = 1 - t / M
            Xnew1 = bestXX * (1 - R)
            Xnew2 = bestXX * (1 + R)
            Xnew1l = bestXX1 * (1 - R)
            Xnew2l = bestXX1 * (1 + R)
            Xnew1r = bestXX2 * (1 - R)
            Xnew2r = bestXX2 * (1 + R)
            Xnew1 = Bounds(Xnew1, lb, ub)
            Xnew2 = Bounds(Xnew2, lb, ub)
            Xnew1l = Bounds(Xnew1l, lb, ub)
            Xnew2l = Bounds(Xnew2l, lb, ub)
            Xnew1r = Bounds(Xnew1r, lb, ub)
            Xnew2r = Bounds(Xnew2r, lb, ub)
            Xnew11 = bestX * (1 - R)
            Xnew22 = bestX * (1 + R)
            Xnew11l = bestX1 * (1 - R)
            Xnew22l = bestX1 * (1 + R)
            Xnew11r = bestX2 * (1 - R)
            Xnew22r = bestX2 * (1 + R)
            Xnew11 = Bounds(Xnew11, lb, ub)
            Xnew22 = Bounds(Xnew22, lb, ub)
            Xnew11l = Bounds(Xnew11l, lb, ub)
            Xnew22l = Bounds(Xnew22l, lb, ub)
            Xnew11r = Bounds(Xnew11r, lb, ub)
            Xnew22r = Bounds(Xnew22r, lb, ub)
            xLB = swapfun(Xnew1)
            xUB = swapfun(Xnew2)
            xLB1 = swapfun(Xnew1l)
            xUB1 = swapfun(Xnew2l)
            xLB2 = swapfun(Xnew1r)
            xUB2 = swapfun(Xnew2r)
            for i in range(pNum + 1, 12):
                X[i, :] = bestXX + (np.random.rand(1, dim)) * (pX[i, :] - Xnew1) + (np.random.rand(1, dim)) * (
                        pX[i, :] - Xnew2)
                X1[i, :] = bestXX1 + (np.random.rand(1, dim)) * (pX1[i, :] - Xnew1l) + (np.random.rand(1, dim)) * (
                        pX1[i, :] - Xnew2l)
                X2[i, :] = bestXX2 + (np.random.rand(1, dim)) * (pX2[i, :] - Xnew1r) + (np.random.rand(1, dim)) * (
                        pX2[i, :] - Xnew2r)
                X[i, :] = Bounds(X[i, :], xLB, xUB)
                X1[i, :] = Bounds(X1[i, :], xLB1, xUB1)
                X2[i, :] = Bounds(X2[i, :], xLB2, xUB2)
                train_bags, train_labels = get_data(posi, nega, channel=1, window_size=501)
                model_type = "CNN"
                loss = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=1,
                                     window_size=501 + 6,
                                     model_file=model_type + '.501', batch_size=batch_size, n_epochs=n_epochs,
                                     num_filters=num_filters, learning_rate=X[i, 0], weight_decay=X[i, 1])
                fit[i, 0] = loss
                train_bags, train_labels = get_data(posi, nega, channel=1, window_size=501)
                model_type = "CNN-LSTM"
                loss1 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=1,
                                      window_size=501 + 6,
                                      model_file=model_type + '.501', batch_size=batch_size, n_epochs=n_epochs,
                                      num_filters=num_filters, learning_rate=X1[i, 0], weight_decay=X1[i, 1])
                fit1[i, 0] = loss1
                train_bags, train_labels = get_data(posi, nega, channel=1, window_size=501)
                model_type = "ResNet"
                loss2 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=1,
                                      window_size=501 + 6,
                                      model_file=model_type + '.501', batch_size=batch_size, n_epochs=n_epochs,
                                      num_filters=num_filters, learning_rate=X2[i, 0], weight_decay=X2[i, 1])
                fit2[i, 0] = loss2
            for i in range(13, 19):
                X[i, :] = pX[i, :] + ((np.random.randn(1)) * (pX[i, :] - Xnew11) + (
                        (np.random.rand(1, dim)) * (pX[i, :] - Xnew22)))
                X1[i, :] = pX1[i, :] + ((np.random.randn(1)) * (pX1[i, :] - Xnew11l) + (
                        (np.random.rand(1, dim)) * (pX1[i, :] - Xnew22l)))
                X2[i, :] = pX2[i, :] + ((np.random.randn(1)) * (pX2[i, :] - Xnew11r) + (
                        (np.random.rand(1, dim)) * (pX2[i, :] - Xnew22r)))
                X[i, :] = Bounds(X[i, :], lb, ub)
                X1[i, :] = Bounds(X1[i, :], lb, ub)
                X2[i, :] = Bounds(X2[i, :], lb, ub)
                train_bags, train_labels = get_data(posi, nega, channel=1, window_size=501)
                model_type = "CNN"
                loss = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=1,
                                     window_size=501 + 6,
                                     model_file=model_type + '.501', batch_size=batch_size, n_epochs=n_epochs,
                                     num_filters=num_filters, learning_rate=X[i, 0], weight_decay=X[i, 1])
                fit[i, 0] = loss
                train_bags, train_labels = get_data(posi, nega, channel=1, window_size=501)
                model_type = "CNN-LSTM"
                loss1 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=1,
                                      window_size=501 + 6,
                                      model_file=model_type + '.501', batch_size=batch_size, n_epochs=n_epochs,
                                      num_filters=num_filters, learning_rate=X1[i, 0], weight_decay=X1[i, 1])
                fit1[i, 0] = loss1
                train_bags, train_labels = get_data(posi, nega, channel=1, window_size=501)
                model_type = "ResNet"
                loss2 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=1,
                                      window_size=501 + 6,
                                      model_file=model_type + '.501', batch_size=batch_size, n_epochs=n_epochs,
                                      num_filters=num_filters, learning_rate=X2[i, 0], weight_decay=X2[i, 1])
                fit2[i, 0] = loss2
            for j in range(20, pop):
                X[j, :] = levy(dim)*bestX + np.random.randn(1, dim) * (np.abs(pX[j, :] - bestXX) + np.abs(pX[j, :] - bestX)) / 2
                X1[j, :] = levy(dim)*bestX1 + np.random.randn(1, dim) * (
                        np.abs(pX1[j, :] - bestXX1) + np.abs(pX1[j, :] - bestX1)) / 2
                X2[j, :] = levy(dim)*bestX2 + np.random.randn(1, dim) * (
                        np.abs(pX2[j, :] - bestXX2) + np.abs(pX2[j, :] - bestX2)) / 2
                X[j, :] = Bounds(X[j, :], lb, ub)
                X1[j, :] = Bounds(X1[j, :], lb, ub)
                X2[j, :] = Bounds(X2[j, :], lb, ub)
                train_bags, train_labels = get_data(posi, nega, channel=1, window_size=501)
                model_type = "CNN"
                loss = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=1,
                                     window_size=501 + 6,
                                     model_file=model_type + '.501', batch_size=batch_size, n_epochs=n_epochs,
                                     num_filters=num_filters, learning_rate=X[j, 0], weight_decay=X[j, 1])
                fit[j, 0] = loss
                train_bags, train_labels = get_data(posi, nega, channel=1, window_size=501)
                model_type = "CNN-LSTM"
                loss1 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=1,
                                      window_size=501 + 6,
                                      model_file=model_type + '.501', batch_size=batch_size, n_epochs=n_epochs,
                                      num_filters=num_filters, learning_rate=X1[j, 0], weight_decay=X1[j, 1])
                fit1[j, 0] = loss1
                train_bags, train_labels = get_data(posi, nega, channel=1, window_size=501)
                model_type = "ResNet"
                loss2 = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=1,
                                      window_size=501 + 6,
                                      model_file=model_type + '.501', batch_size=batch_size, n_epochs=n_epochs,
                                      num_filters=num_filters, learning_rate=X2[j, 0], weight_decay=X2[j, 1])
                fit2[j, 0] = loss2
            XX = pX
            XX1 = pX1
            XX2 = pX2
            for i in range(pop):
                if fit[i, 0] < pFit[i, 0]:
                    pFit[i, 0] = fit[i, 0]
                    pX[i, :] = X[i, :]
                if pFit[i, 0] < fMin:
                    fMin = pFit[i, 0]
                    bestX = pX[i, :]
            for i in range(pop):
                if fit1[i, 0] < pFit1[i, 0]:
                    pFit1[i, 0] = fit1[i, 0]
                    pX1[i, :] = X1[i, :]
                if pFit1[i, 0] < fMin1:
                    fMin1 = pFit1[i, 0]
                    bestX1 = pX1[i, :]
            for i in range(pop):
                if fit2[i, 0] < pFit2[i, 0]:
                    pFit2[i, 0] = fit2[i, 0]
                    pX2[i, :] = X2[i, :]
                if pFit2[i, 0] < fMin2:
                    fMin2 = pFit2[i, 0]
                    bestX2 = pX2[i, :]
            X = pX
            X1 = pX1
            X2 = pX2

        print ("501")
        train_bags, train_labels = get_data(posi, nega, channel = 1, window_size = 501)
        model_type = "CNN"
        train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 1, window_size = 501 + 6, model_file = model_type + '.501', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters, learning_rate=bestX[0], weight_decay=bestX[1])
        model_type = "CNN-LSTM"
        train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 1, window_size = 501 + 6, model_file = model_type + '.501', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters, learning_rate=bestX1[0], weight_decay=bestX1[1])
        model_type = "ResNet"
        train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 1, window_size = 501 + 6, model_file = model_type + '.501', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters, learning_rate=bestX2[0], weight_decay=bestX2[1])

    elif predict:
        model_type = "CNN"
        X_test, X_labels = get_data(testfile, nega , channel = 7, window_size = 101)
        CnnPre1 = predict_network(model_type, np.array(X_test), channel = 7, window_size = 101 + 6, model_file = model_type+ '.101', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        X_test, X_labels = get_data(testfile, nega , channel = 4, window_size = 151)
        CnnPre2 = predict_network(model_type, np.array(X_test), channel = 4, window_size = 151 + 6, model_file = model_type+ '.151', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        X_test, X_labels = get_data(testfile, nega , channel = 3, window_size = 201)
        CnnPre3 = predict_network(model_type, np.array(X_test), channel = 3, window_size = 201 + 6, model_file = model_type+ '.201', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        X_test, X_labels = get_data(testfile, nega , channel = 2, window_size = 251)
        CnnPre4 = predict_network(model_type, np.array(X_test), channel = 2, window_size = 251 + 6, model_file = model_type+ '.251', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        X_test, X_labels = get_data(testfile, nega , channel = 2, window_size = 301)
        CnnPre5 = predict_network(model_type, np.array(X_test), channel = 2, window_size = 301 + 6, model_file = model_type+ '.301', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        X_test, X_labels = get_data(testfile, nega , channel = 2, window_size = 351)
        CnnPre6 = predict_network(model_type, np.array(X_test), channel = 2, window_size = 351 + 6, model_file = model_type+ '.351', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        X_test, X_labels = get_data(testfile, nega , channel = 1, window_size = 401)
        CnnPre7 = predict_network(model_type, np.array(X_test), channel = 1, window_size = 401 + 6, model_file = model_type+ '.401', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        X_test, X_labels = get_data(testfile, nega , channel = 1, window_size = 451)
        CnnPre8 = predict_network(model_type, np.array(X_test), channel = 1, window_size = 451 + 6, model_file = model_type+ '.451', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        X_test, X_labels = get_data(testfile, nega , channel = 1, window_size = 501)
        CnnPre9 = predict_network(model_type, np.array(X_test), channel = 1, window_size = 501 + 6, model_file = model_type+ '.501', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)

        model_type = "CNN-LSTM"
        X_test, X_labels = get_data(testfile, nega , channel = 7, window_size = 101)
        CnnLstmPre1 = predict_network(model_type, np.array(X_test), channel = 7, window_size = 101 + 6, model_file = model_type+ '.101', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        X_test, X_labels = get_data(testfile, nega , channel = 4, window_size = 151)
        CnnLstmPre2 = predict_network(model_type, np.array(X_test), channel = 4, window_size = 151 + 6, model_file = model_type+ '.151', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        X_test, X_labels = get_data(testfile, nega , channel = 3, window_size = 201)
        CnnLstmPre3 = predict_network(model_type, np.array(X_test), channel = 3, window_size = 201 + 6, model_file = model_type+ '.201', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        X_test, X_labels = get_data(testfile, nega , channel = 2, window_size = 251)
        CnnLstmPre4 = predict_network(model_type, np.array(X_test), channel = 2, window_size = 251 + 6, model_file = model_type+ '.251', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        X_test, X_labels = get_data(testfile, nega , channel = 2, window_size = 301)
        CnnLstmPre5 = predict_network(model_type, np.array(X_test), channel = 2, window_size = 301 + 6, model_file = model_type+ '.301', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        X_test, X_labels = get_data(testfile, nega , channel = 2, window_size = 351)
        CnnLstmPre6 = predict_network(model_type, np.array(X_test), channel = 2, window_size = 351 + 6, model_file = model_type+ '.351', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        X_test, X_labels = get_data(testfile, nega , channel = 1, window_size = 401)
        CnnLstmPre7 = predict_network(model_type, np.array(X_test), channel = 1, window_size = 401 + 6, model_file = model_type+ '.401', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        X_test, X_labels = get_data(testfile, nega , channel = 1, window_size = 451)
        CnnLstmPre8 = predict_network(model_type, np.array(X_test), channel = 1, window_size = 451 + 6, model_file = model_type+ '.451', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        X_test, X_labels = get_data(testfile, nega , channel = 1, window_size = 501)
        CnnLstmPre9 = predict_network(model_type, np.array(X_test), channel = 1, window_size = 501 + 6, model_file = model_type+ '.501', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)

        model_type = "ResNet"
        X_test, X_labels = get_data(testfile, nega , channel = 7, window_size = 101)
        ResNetPre1 = predict_network(model_type, np.array(X_test), channel = 7, window_size = 101 + 6, model_file = model_type+ '.101', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        X_test, X_labels = get_data(testfile, nega , channel = 4, window_size = 151)
        ResNetPre2 = predict_network(model_type, np.array(X_test), channel = 4, window_size = 151 + 6, model_file = model_type+ '.151', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        X_test, X_labels = get_data(testfile, nega , channel = 3, window_size = 201)
        ResNetPre3 = predict_network(model_type, np.array(X_test), channel = 3, window_size = 201 + 6, model_file = model_type+ '.201', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        X_test, X_labels = get_data(testfile, nega , channel = 2, window_size = 251)
        ResNetPre4 = predict_network(model_type, np.array(X_test), channel = 2, window_size = 251 + 6, model_file = model_type+ '.251', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        X_test, X_labels = get_data(testfile, nega , channel = 2, window_size = 301)
        ResNetPre5 = predict_network(model_type, np.array(X_test), channel = 2, window_size = 301 + 6, model_file = model_type+ '.301', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        X_test, X_labels = get_data(testfile, nega , channel = 2, window_size = 351)
        ResNetPre6 = predict_network(model_type, np.array(X_test), channel = 2, window_size = 351 + 6, model_file = model_type+ '.351', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        X_test, X_labels = get_data(testfile, nega , channel = 1, window_size = 401)
        ResNetPre7 = predict_network(model_type, np.array(X_test), channel = 1, window_size = 401 + 6, model_file = model_type+ '.401', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        X_test, X_labels = get_data(testfile, nega , channel = 1, window_size = 451)
        ResNetPre8 = predict_network(model_type, np.array(X_test), channel = 1, window_size = 451 + 6, model_file = model_type+ '.451', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        X_test, X_labels = get_data(testfile, nega , channel = 1, window_size = 501)
        ResNetPre9 = predict_network(model_type, np.array(X_test), channel = 1, window_size = 501 + 6, model_file = model_type+ '.501', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)

        CnnPre = (CnnPre1 + CnnPre2 + CnnPre3 + CnnPre4 + CnnPre5 + CnnPre6 + CnnPre7 + CnnPre8 + CnnPre9) / 9
        CnnLstmPre = (CnnLstmPre1 + CnnLstmPre2 + CnnLstmPre3 + CnnLstmPre4 + CnnLstmPre5 + CnnLstmPre6 + CnnLstmPre7 + CnnLstmPre8 + CnnLstmPre9) / 9
        ResNetPre = (ResNetPre1 + ResNetPre2 + ResNetPre3 + ResNetPre4 + ResNetPre5 + ResNetPre6 + ResNetPre7 + ResNetPre8 + ResNetPre9) / 9


        fMin, bestX, bestY = DBO.DBO(CnnPre, CnnLstmPre, ResNetPre, X_labels)
        print('AUC' + str(-fMin) + '\n')


    elif motif:
        motif_seqs = []
        data = read_data_file(posi, nega)
        motif_seqs = data['seq']
        if posi == '' or nega == '':
            print ('To identify motifs, you need training positive and negative sequences using global CNNs.')
        train_bags, train_labels = get_data(posi, nega, channel = 1, window_size = 501)
        train_network("CNN", np.array(train_bags), np.array(train_labels), channel = 1, window_size = 501 + 6, model_file = model_file + '.501', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters, motif = motif, motif_seqs = motif_seqs, motif_outdir = motif_outdir)
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
    parser.add_argument('--batch_size', type=int, default=4096, help='The size of a single mini-batch (default value: 4096)')
    parser.add_argument('--num_filters', type=int, default=16, help='The number of filters for CNNs (default value: 16)')
    parser.add_argument('--n_epochs', type=int, default=30, help='The number of training epochs (default value: 30)')
    parser.add_argument('--motif', type=bool, default=False, help='It is used to identify binding motifs from sequences.')
    parser.add_argument('--motif_dir', type=str, default='motifs', help='The dir used to store the prediction binding motifs.')
    args = parser.parse_args()
    return args

parser = argparse.ArgumentParser()
args = parse_arguments(parser)
print (args)
run(args)
