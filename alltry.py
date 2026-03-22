import numpy as np
import torch 
from   torch.utils.data  import Dataset,DataLoader
from obspy import read 
device = torch.device('cuda')
rection = read('real_test.sgy')
data  =np.stack([tr.data for tr in rection ])
data_max = np.max(data)
data_norm = data/data_max
class SeismicDataset(Dataset):
    def __init__(self,data_list,patch_size=128,):
        super().__init__()
        self.data = data_list
        self.patch_size = patch_size
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        trace = self.data[idx]
        start = np.random.randint(0,len(self.data)-self.patch_size+1)
        patch = trace[start:start+self.patch_size]
        mask =np.random.choice([0,1],patch.shape,[0.99,0.01])
        incomplete_patch = mask * patch
        x  =  torch.from_numpy(incomplete_patch.astype(np.float32)).unsqueeze(0)
        y  =  torch.from_numpy(patch.astype(np.float32)).unsqueeze(0)
        return x ,y

dataset = SeismicDataset(data_norm,patch_size=128)
train_loader = DataLoader(dataset,batch_size=32,shuffle=True)
import torch.nn as nn 
import torch.optim as optim 
class SeismicCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.CNN = nn.Sequential(nn.Conv1d(in_channels=1,out_channels=16,kernel_size=3,padding=1 ),
                                nn.ReLU(),
                                nn.Conv1d(in_channels=16,out_channels=16,kernel_size=3,padding=1 ),
                                nn.ReLU(),
                                nn.Conv1d(in_channels=16,out_channels=1,kernel_size=3,padding=1 ))
    def forward(self,x):
        x = self.CNN(x)
        return x 
creterion = nn.MSELoss()
model = SeismicCNN().to(device)
optimizer = optim.Adam(model.parameters(),lr =0.01)
for epoch  in range(100):
    total_loss = 0
    for batch_idx,(inputs,labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels =labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = creterion(outputs,labels)
        loss.backward()
        optimizer.step()
        total_loss +=loss.item()
        if (epoch+1) %5 == 0 :
            print(f'Epoch:[{epoch+1}/100], Avg Loss :{total_loss/len(train_loader)}')


origin = labels[0].detach().cpu().squeeze().numpy()
incomplete = inputs[0].detach().cpu().squeeze().numpy()
output   = outputs[0].detach().cpu().squeeze().numpy()
import matplotlib.pyplot as plt 

plt.figure(figsize=(12,10))

plt.subplot(3,1,1)
plt.plot(origin,color ='green')
plt.title('Origin')
plt.grid(True)

plt.subplot(3,1,2)
plt.plot(incomplete,color = 'red')
plt.title('Incomplete')
plt.grid(True)

plt.subplot(3,1,3)
plt.plot(output,color='blue')
plt.title('Output')
plt.grid(True)

plt.show()
plt.savefig('终于搓出来了.png')