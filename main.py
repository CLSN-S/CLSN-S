import Dataset_add
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from scipy import stats
from torch.autograd import Variable
from scipy.signal import argrelextrema
from scipy.signal import find_peaks
import numpy as np
import math
import os
import torch.nn.functional as F
import torch.nn.utils

root_raw_train = r'D:/railyway/文章二/data_set/raw_train.txt'
root_ref_train = r'D:/railyway/文章二/data_set/ref_train.txt'
root_raw_test = r'D:/railyway/文章二/data_set/raw_test.txt'
root_ref_test = r'D:/railyway/文章二/data_set/ref_test.txt'
train_batch_size = 64
test_batch_size = 64
train_dataset = Dataset_add.MyDataset(root_raw_train, root_ref_train)
test_dataset = Dataset_add.MyDataset(root_raw_test, root_ref_test)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=True)


def count_parameters_using_vector(model):
    params_vector = torch.nn.utils.parameters_to_vector(model.parameters())
    return len(params_vector)

class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.attention_weights = nn.Parameter(torch.randn(hidden_size))

    def forward(self, inputs):
        # inputs shape: [batch_size, seq_len, hidden_size]
        scores = torch.tanh(inputs @ self.attention_weights)
        attention_weights = F.softmax(scores, dim=1)
        # apply weights
        weighted_average = inputs * attention_weights.unsqueeze(-1)
        return weighted_average

class MY_NET(nn.Module):
    def __init__(self):
        super(MY_NET, self).__init__()
        self.input_size = 16
        self.seq_length = 16
        self.num_layers = 2
        self.hidden_size = 128
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv1d(1, 256, kernel_size=100, stride=1,  padding='same'),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.2)
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv1d(256, 128, kernel_size=100, stride=2,  padding=49),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.2)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv1d(128, 1, kernel_size=100, stride=2,  padding=49),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.2)
        )
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True, dropout = 0.2)
        self.attention = SelfAttention(self.hidden_size * 2)
        self.fc1 = nn.Linear(self.hidden_size * 2*self.seq_length, 1024)  # 2 for bidirection
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 1024)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(-1, self.seq_length, self.input_size)
        # Set initial states
        h0 = torch.randn(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # 2 for bidirection
        c0 = torch.randn(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        # Forward propagate LSTM
        out, _ = self.lstm(out, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        # Decode the hidden state of the last time step
        # Attention layer
        out = self.attention(out)
        out = F.relu(self.fc1(out.reshape(out.size()[0], -1))) #用所有层
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        out = out.unsqueeze(1)
        return out

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.input_size = 64
        self.seq_length = 16
        self.num_layers = 1
        self.hidden_size = 512
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=False, dropout = 0.2)
        self.fc1 = nn.Linear(self.hidden_size *self.seq_length, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 1024)

    def forward(self, x):
        x = x.reshape(-1, self.seq_length, self.input_size)
        # Set initial states
        h0 = torch.randn(self.num_layers, x.size(0), self.hidden_size).to(x.device)  # 2 for bidirection
        c0 = torch.randn(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        # Decode the hidden state of the last time step
        out = F.relu(self.fc1(out.reshape(out.size()[0], -1))) #用所有层
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        out = out.unsqueeze(1)
        return out

class BiLSTM(nn.Module):
    def __init__(self):
        super(BiLSTM, self).__init__()
        self.input_size = 64
        self.seq_length = 16
        self.num_layers = 2
        self.hidden_size = 512
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True, dropout = 0.2)
        self.fc1 = nn.Linear(self.hidden_size * 2*self.seq_length, 1024)  # 2 for bidirection
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 1024)

    def forward(self, x):
        x = x.reshape(-1, self.seq_length, self.input_size)

        # Set initial states
        h0 = torch.randn(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # 2 for bidirection
        c0 = torch.randn(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        # Decode the hidden state of the last time step
        # out = F.relu(self.fc1(out[:, -1, :]))  #只用最后一层
        out = F.relu(self.fc1(out.reshape(out.size()[0], -1))) #用所有层

        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        out = out.unsqueeze(1)
        # print(out.size())
        return out

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv1d(1, 256, kernel_size=100, stride=1,  padding='same'),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(inplace=True),
            # torch.nn.BatchNorm1d(128),
            torch.nn.Dropout(0)
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv1d(256, 128, kernel_size=100, stride=1,  padding='same'),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(inplace=True),
            # torch.nn.BatchNorm1d(128),
            torch.nn.Dropout(0)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv1d(128, 32, kernel_size=100, stride=1,  padding='same'),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(inplace=True),
            # torch.nn.BatchNorm1d(128),
            torch.nn.Dropout(0)
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv1d(32, 1, kernel_size=100, stride=1, padding='same'),
            # torch.nn.ReLU(inplace=True)
        )

    def forward(self, net_data):
        net_data = self.layer1(net_data)
        net_data = self.layer2(net_data)
        net_data = self.layer3(net_data)
        net_data = self.layer4(net_data)
        return net_data

def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_dataloader, 0):
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)  # 将数据放在GPU上跑所需要的代码
        optimizer.zero_grad()
        # 前馈+反馈+更新
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % 50 == 0:
            training_loss.append(running_loss / 50)
            print('[%d,%5d] loss:%.3f' % (epoch + 1, batch_idx + 1, running_loss / 50))
            running_loss = 0.0


def test():
    with torch.no_grad():  # 下面的代码就不会再计算梯度
        global real_sum, detect_sum, cdf
        testing_coefficient = []
        real_sum, detect_sum = 0, 0
        num_histogram = np.zeros(100)
        cdf = np.zeros(100)
        sum_idx = 0
        for batch_idx, data in enumerate(test_dataloader, 0):
            inputs, target = data
            raw_data = inputs[10, 0, :].numpy()
            ref_data = target[10, 0, :].numpy()
            inputs, target = inputs.to(device), target.to(device)  # 将数据放在GPU上跑所需要的代码
            outputs = model(inputs)
            #展示模型输出结果
            # out_data = outputs[10, 0, :].cpu().numpy()
            # #寻找峰值点
            # out_peaks, _ = find_peaks(out_data, height=out_data.max() / 2, distance=5)
            # ref_peaks, _ = find_peaks(ref_data, height=ref_data.max() / 2, distance=5)
            # #绘图
            # x_axis = np.array(range(1, 1025))
            # plt.figure()
            # plt.scatter(out_peaks, out_data[out_peaks], color='red', s=100, label='Detected Fasteners')
            # plt.scatter(ref_peaks, ref_data[ref_peaks], color='green', s=100, label='Ref Fasteners')
            # plt.plot(x_axis, raw_data, c='b', lw=1, label='Input')
            # plt.plot(x_axis, ref_data, c='g', lw=2, label='Reference')
            # plt.plot(x_axis, out_data, c='r', lw=2, label='Output')
            # plt.grid(True)
            # plt.legend()
            # plt.show()
            for idx in range(outputs.size()[0]):
                sum_idx = sum_idx + 1
                #求相关系数
                temp_coeff = stats.pearsonr(outputs[idx, 0, :].cpu(), target[idx,0, :].cpu())[0]
                if np.isnan(temp_coeff):
                    temp_coeff = 0.0000001
                #展示垃圾数据
                # if temp_coeff < 0.9:
                #     print(temp_coeff)
                #     plt.figure()
                #     plt.plot(inputs[idx, 0, :].cpu().numpy(), c='b', lw=1, label='Input')
                #     plt.plot(target[idx, 0, :].cpu().numpy(), c='g', lw=2, label='Reference')
                #     plt.plot(outputs[idx, 0, :].cpu().numpy(), c='r', lw=2, label='Output')
                #     plt.grid(True)
                #     plt.legend()
                #     plt.show()
                testing_coefficient.append(temp_coeff)
                num_histogram[math.floor(temp_coeff*100)] = num_histogram[math.floor(temp_coeff*100)]+1
                #求扣件个数
                # Convert the numpy array to a PyTorch Tensor
                detect_indices = torch.from_numpy(argrelextrema(outputs[idx, 0, :].cpu().numpy(), np.greater, order=5)[0])
                target_indices = torch.from_numpy(argrelextrema(target[idx, 0, :].cpu().numpy(), np.greater, order=5)[0])
                # Now use these tensors as indices
                detect_peaks = outputs[idx, 0, :].take(detect_indices.to(device))
                target_peaks = target[idx, 0, :].take(target_indices.to(device))
                # 过滤掉低于阈值的峰值点
                filtered_detect_peaks = detect_peaks[detect_peaks.cpu().numpy() > outputs[idx, 0, :].cpu().numpy().max()/ 2]
                filtered_target_peaks = target_peaks[target_peaks.cpu().numpy() > target[idx, 0, :].cpu().numpy().max() / 2]
                detect_sum = detect_sum + len(filtered_detect_peaks)
                real_sum = real_sum + len(filtered_target_peaks)
        tem_sum = 0
        for idx in range(len(cdf)):
            tem_sum = tem_sum + num_histogram[idx]
            cdf[idx] = tem_sum/sum_idx
        # plt.plot(cdf)
        # plt.show()


if __name__ == '__main__':
    #模型选择
    model_select = 'MY_NET'
    # 训练模型标志
    train_flag = 0  # 1-训练  0-测试

    if model_select == 'CNN':
        model = CNN()
    elif model_select == 'BiLSTM':
        model = BiLSTM()
    elif model_select == 'LSTM':
        model = LSTM()
    elif model_select == 'MY_NET':
        model = MY_NET()
    ## 打印参数数量
    total_params = count_parameters_using_vector(model)
    print(f"Total parameters: {total_params}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    #损失函数
    criterion = torch.nn.MSELoss()
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # 训练结果
    training_loss = []

    # 训练
    if train_flag == 1:
        for epoch in range(50):
            train(epoch)
            model_path = os.path.join('model', model_select + '_' + str(epoch) + '.pkl')
            torch.save(model.state_dict(), model_path)
        #打印 training_loss
        with open(model_select+'.txt', 'a') as f:
            f.write('\ntraining_loss\n')
            np.savetxt(f, training_loss, fmt='%.3f', newline=' ')
        plt.figure(1)
        plt.plot(training_loss)
        plt.title('training_loss')
        plt.legend(loc=1)
        plt.show()
    # 测试
    else:
        for i in range(30):
            model_num = 24
            model_path = os.path.join('model', model_select + '_' + str(model_num)+'.pkl')
            model.load_state_dict(torch.load(model_path))
            model.eval()
            test()
            with open(model_select+'.txt', 'a') as f:
                f.write(f'\n\n\nmodel_num={model_num}')
                f.write('\ndetect_sum    real_sum    error\n')
                np.savetxt(f, [detect_sum, real_sum], fmt='%d', newline=' ')
                f.write(str((detect_sum - real_sum) / real_sum))
                f.write('\ncdf\n')
                np.savetxt(f, cdf, fmt='%.2f', newline=' ')