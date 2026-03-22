from obspy import read
import numpy as np 
from torch.utils.data import Dataset
import torch
import  torch.nn as nn
##数据处理
#读取数据
rection = read('real_test.sgy')
data = np.stack([tr.data for tr in rection])
#归一化
data_max =np.abs(np.max(data))
data_norm = data/data_max
#传入Dataset
class SeismicDataset(Dataset):
    def __init__(self,data_list):
        super(SeismicDataset,self).__init__()
        self.data = data_list
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx ):
        trace = self.data[idx]
        mask = np.random.choice([0,1],trace.shape,[0.7,0.3])
        incomplete_data = trace *mask 
        x =torch.from_numpy(incomplete_data.astype(np.float32))
        y=torch.from_numpy(trace.astype(np.float32))
        return x, y
tensor_data = SeismicDataset(data_norm)

##神经网络部分
#定义CNN模型
class SeismicNN(nn.Module):
    def __init__(self):
        super(SeismicNN,self).__init__()
        self.layer = nn.Sequential(nn.Conv1d(in_channels=1,out_channels=16,kernel_size=3,padding=1),
                                   nn.ReLU(),
                                   nn.Conv1d(in_channels=16,out_channels=16,kernel_size=3,padding= 1 ),
                                   nn.ReLU(),
                                   nn.Conv1d(in_channels=16,out_channels=1,kernel_size=3,padding=1))
    def forward(self,x):
        x = self.layer(x)
        return x
import torch.optim as optim 

device = torch.device("cuda")
model =SeismicNN().to(device)
creterion = nn.MSELoss()
optimizer= optim.Adam(model.parameters(),lr= 0.01)
input_data ,label_data =tensor_data[0] 
input_data = input_data.unsqueeze(0).unsqueeze(0).to(device)
label_data  =label_data.unsqueeze(0).unsqueeze(0).to(device)
for epoch in range(100):    
    optimizer.zero_grad()
    prediction =model(input_data)
    loss =creterion(prediction,label_data)
    loss.backward()
    optimizer.step()
    print(f'已玩完成第{epoch+1}次训练，误差为{loss.item()}')


original =label_data.detach().cpu().numpy().flatten()
incomplete =input_data.detach().cpu().numpy().flatten()
output = prediction.detach().cpu().numpy().flatten()
import matplotlib.pyplot as plt 

plt.figure(figsize=(12,10))
plt.subplot(3,1,1)
plt.plot(incomplete,color = 'blue')
plt.title('Incomplete')
plt.grid(True)

plt.subplot(3,1,2)
plt.plot(output,color ='red')
plt.title('Output')
plt.grid(True)
plt.subplot(3,1,3)
plt.plot(original,color ='green')
plt.title("Original")
plt.grid(True)
plt.tight_layout()
plt.show
plt.savefig('手搓的.png')

