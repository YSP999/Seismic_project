import torch
import numpy as np
import matplotlib.pyplot as plt 
from obspy import read 
## 数据处理
rection = read('real_test.sgy')
data = np.stack([tr.data for tr  in rection ])
data_max = np.max(data )
data_norm = data/data_max

from torch.utils.data import DataLoader,Dataset,random_split
#封装dataset
class SeismicDataset(Dataset):
    def __init__(self,data_list,patch_size =128):
        super().__init__()
        self.data = data_list
        self.patch_size = patch_size
    def __len__(self):
        return len(self.data )
    def __getitem__(self,idx):
        trace = self.data[idx]
        start = np.random.randint(0,len(trace)-self.patch_size+1)
        patch = trace[start:start+self.patch_size]
        mask  =np.random.choice([0,1],patch.shape,[0.9,0.1])
        incomplete_patch = patch *mask 
        x = torch.from_numpy(incomplete_patch.astype(np.float32)).unsqueeze(0)
        y = torch.from_numpy(patch.astype(np.float32)).unsqueeze(0)
        return x,y
device = torch.device('cuda')


def draw_comparison(inputs, outputs, labels, save_name):
    # 取 Batch 中的第一个样本
    in_np = inputs.detach().cpu().numpy()[0, 0, :]
    out_np = outputs.detach().cpu().numpy()[0, 0, :]
    gt_np = labels.detach().cpu().numpy()[0, 0, :]
    
    plt.figure(figsize=(12, 10))
    plt.subplot(3, 1, 1); plt.plot(gt_np, color='green'); plt.title('Ground Truth (Origin)')
    plt.subplot(3, 1, 2); plt.plot(in_np, color='red'); plt.title('Input (Incomplete)')
    plt.subplot(3, 1, 3); plt.plot(out_np, color='blue'); plt.title('Model Output (Reconstructed)')
    
    plt.tight_layout()
    plt.savefig(save_name) # 根据传入的名字保存
    plt.close()

full_dataset = SeismicDataset(data_norm,patch_size=128)

train_size =int(0.8*len(full_dataset))
test_size = len(full_dataset) -train_size

train_dataset ,test_dataset =random_split(full_dataset,[train_size,test_size])

train_loader =DataLoader(train_dataset,batch_size=32,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=32,shuffle=False)
import torch.nn as nn 
class SeismicCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.CNN = nn.Sequential(nn.Conv1d(in_channels=1,out_channels=16,kernel_size=3,padding=1),
                                 nn.ReLU(),
                                 nn.Conv1d(in_channels=16,out_channels=16,kernel_size=3,padding=1),
                                 nn.ReLU(),
                                 nn.Conv1d(in_channels=16,out_channels=1,kernel_size=3,padding=1))
    def forward(self,x):
        x = self.CNN(x)
        return x
model = SeismicCNN().to(device)
import torch.optim as optim 
optimizer =optim.Adam(model.parameters(),lr = 0.01)
creterion = nn.MSELoss()
for epoch in range(100):
    total_loss =0
    for bateh_idx,(inputs,labels) in enumerate(train_loader):
        train_inputs = inputs.to(device)
        train_labels = labels.to(device)
        optimizer.zero_grad()
        train_outputs = model(train_inputs)
        loss = creterion(train_outputs,train_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch+1)% 5  == 0 :
        print(f'Epoch:[{epoch+1}/100],Avg Loss:{total_loss/len(train_loader)}')

    if (epoch+1) %20 == 0:
        draw_comparison(train_inputs,train_outputs,train_labels,f'第{epoch+1}对比图')





model.eval()
test_loss =0 
with torch.no_grad():
    for batch_size,(inputs,labels)  in  enumerate(test_loader):
        test_inputs = inputs.to(device)
        test_labels = labels.to(device)
        test_outputs = model(test_inputs)
        loss = creterion(test_inputs,test_labels)
        test_loss +=loss.item()
avg_test_loss = test_loss/len(test_loader)
print(f'Test Loss :{avg_test_loss}')
draw_comparison(test_inputs, test_outputs, test_labels,'Final Test.png')

# input = inputs.detach().cpu().numpy()[0,0,:]
# output = outputs.detach().cpu().numpy()[0,0,:]
# origin = labels.detach().cpu().numpy()[0,0,:]


# print(input.shape)

# plt.figure(figsize=(12,10))
# plt.subplot(3,1,1)
# plt.plot(origin,color ='green')
# plt.title('Origin')
# plt.grid(True)


# plt.subplot(3,1,2)
# plt.plot(input,color= 'red')
# plt.title('Incomplete')
# plt.grid(True)

# plt.subplot(3,1,3)
# plt.plot(output,color= 'blue')
# plt.title('Output')
# plt.grid(True)
# plt.show()
# plt.savefig('早上手搓的对比图.png')


fk = np.fft.fft2(data_norm)
fk_shift  =np.fft.fftshift(fk)

fk_log = np.log(20*fk_shift+1e-6)
fk_abs = np.abs(fk_log)
plt.figure(figsize=(12,8))
im = plt.imshow(fk_abs,cmap='gray',aspect='auto')
plt.colorbar(im,label='Amtitude')
plt.show()
plt.grid(True)
plt.savefig('早上手搓的灰度图.png')