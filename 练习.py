##数据预处理
from obspy import read 
import numpy as np
rection = read('real_test.sgy')
data = np.stack([tr.data for tr in rection])
data_max = np.abs(np.max(data))
data_norm = data/data_max
import torch
from torch.utils.data import Dataset
class SeismicDataset(Dataset):
    def __init__(self,data_list):
        super().__init__()
        self.data = data_list
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        trace = self.data[idx]
        mask = np.random.choice([0,1],trace.shape,[0.7,0.3])
        incomplete_data =trace *mask 
        x = torch.from_numpy(incomplete_data.astype(np.float32))
        y = torch.from_numpy(trace.astype(np.float32))
        return x,y 
##神经网路部分
import torch.nn as nn
import torch.optim as optim 
device = torch.device('cuda')
class SeismicCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.CNN = nn.Sequential(nn.Conv1d(in_channels=1,out_channels=16,kernel_size=3,padding=1),
                                 nn.ReLU(),
                                 nn.Conv1d(in_channels=16,out_channels=16,kernel_size=3,padding =1 ),
                                 nn.ReLU(),
                                 nn.Conv1d(in_channels=16,out_channels=1,kernel_size=3,padding = 1))
    def forward(self,x):
        x= self.CNN(x)
        return x
model = SeismicCNN().to(device)
input_data ,label_data = SeismicDataset(data_norm)[0]
input_data = input_data.unsqueeze(0).unsqueeze(0).to(device)
label_data =label_data.unsqueeze(0).unsqueeze(0).to(device)
creterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr =0.01)
for epoch in range (100):
    optimizer.zero_grad()
    output_data =model(input_data)
    loss =creterion(output_data,label_data)
    loss.backward()
    optimizer.step()
    print(f'第{epoch+1}次训练完成，其误差为{loss.item():6f}')
##对输出结果numpy 化
original = label_data.detach().cpu().numpy().flatten()
output =output_data.detach().cpu().numpy().flatten()
incomplete = input_data.detach().cpu().numpy().flatten()


import matplotlib.pyplot as plt 
plt.figure(figsize=(12,10))

plt.subplot(3,1,1)
plt.plot(incomplete,color = 'red')
plt.title('Incomplete')
plt.grid(True)

plt.subplot(3,1,2)
plt.plot(output,color= 'blue')
plt.title('Output')
plt.grid(True)

plt.subplot(3,1,3)
plt.plot(original,color='green')
plt.title('Origin')
plt.grid(True)

plt.show()
plt.savefig('一定要成功啊.png')