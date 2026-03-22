import numpy as np
import torch 
from torch.utils.data import DataLoader,Dataset
import torch.nn as nn 
from obspy import read
import torch.optim as optim
class SeismicDataset(Dataset):
    def __init__(self,data_list,patch_size =128):
        super().__init__()
        self.data = data_list
        self.path_size  = patch_size
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        trace = self.data[idx]
        start = np.random.randint(0,len(self.data)-self.path_size+1)
        patch_data = trace[start:start+self.path_size]
        mask = np.random.choice([0,1],self.path_size,[0.7,0.3])
        patch = mask*patch_data
        x =torch.from_numpy(patch.astype(np.float32)).unsqueeze(0).to(device)
        y =torch.from_numpy(patch_data.astype(np.float32)).unsqueeze(0).to(device)
        return x ,y 
##数据处理
#读取数据
rection = read('real_test.sgy')
data = np.stack([tr.data for tr in rection])
#归一化
data_max =np.abs(np.max(data))
data_norm = data/data_max
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

dataset =SeismicDataset(data_norm,patch_size=128)
train_loader = DataLoader(dataset,batch_size= 32,shuffle=True)
device = torch.device('cuda')
model = SeismicNN().to(device)
optimizer = optim.Adam(model.parameters(),lr =0.01)
creterion = nn.MSELoss()
for epoch in range(100 ):
    total_loss  =0 
    for batch_idx, (inputs,labels) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(inputs)
        loss = creterion(output,labels)
        loss.backward()
        optimizer.step()
        total_loss +=loss.item()
    if (epoch+1) % 5  == 0:
        print(f'Epoch[{epoch+1}/100],Avg Loss{total_loss/len(train_loader):6f}')
      

     