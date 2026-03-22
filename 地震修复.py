import torch 
import numpy as np
from obspy import read 
rection = read('real_test.sgy')
data =  np.stack([tr.data for tr in rection ])
#print(data[:5])

##归一化
data_max = np.abs(np.max(data))
data_norm = data/data_max
# print(f'归一化后最大值{np.max(data_norm)},最小值{np.min(data_norm)}')

# data_tensor =  torch.from_numpy(data_norm.astype(np.float32))
# print(data_tensor.shape)
# print(len(data_tensor))

from  torch.utils.data  import Dataset
class SeismicDataset(Dataset):
    def __init__(self,data_list):
        super(SeismicDataset,self).__init__()
        self.data = data_list
    def __len__(self):
        return(len(self.data))
    def __getitem__(self, idx):
        print(f"我正在帮你处理第{idx+1}条地震道")
        original_data  =self.data[idx]
        mask = np.random.choice([0,1],original_data.shape,[0.7,0.3])
        incomplete_data =mask*original_data
        x = torch.from_numpy(incomplete_data.astype(np.float32))
        y = torch.from_numpy(original_data.astype(np.float32))
        return x, y
    
tensor_data = SeismicDataset(data_norm)
print(f'一共{len(tensor_data)}条地震道')

# first_trace = tensor_data[0]

import torch.nn as nn
class SeismicConv(nn.Module):
    def __init__(self):
        super(SeismicConv,self).__init__()
        self.conv = nn.Sequential(nn.Conv1d(in_channels=1,out_channels=16,kernel_size=3,padding =1),
                                  nn.ReLU(),
                                  nn.Conv1d(in_channels=16,out_channels=16,kernel_size=3,padding =1),
                                  nn.ReLU(),
                                  nn.Conv1d(in_channels=16,out_channels=1,kernel_size=3,padding =1))
    def forward(self,x):
        x =self.conv(x)
        return x

model = SeismicConv()
# input_data ,label_data = tensor_data[0]
# input_batch =input_data.unsqueeze(0)
# output_batch =model(input_batch)
# print(f'输入形状{input_batch.shape},输出形状{output_batch.shape}')


import torch.optim as optim 
creterion =  nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr =0.01)
input_data ,label_data = tensor_data[0]
input_batch = input_data.unsqueeze(0)
label_batch = label_data.unsqueeze(0)
print(input_batch.shape)
for epoch in range(100):
    optimizer.zero_grad()
    output_batch = model(input_batch)
    loss =creterion(output_batch,label_batch)
    loss.backward()
    optimizer.step()
    print(f'第{epoch+1}次训练已经完成，其误差为{loss.item()}')


import matplotlib.pyplot as plt 
original_data  =label_batch.detach().cpu().numpy().flatten()
output_data =  output_batch.detach().cpu().numpy().flatten()
incomplete_data = input_batch.detach().cpu().numpy().flatten()

plt.figure(figsize=(12,10))
plt.subplot(3,1,3)
plt.plot(original_data,color='green')
plt.title('Original Data')

plt.subplot(3,1,2)
plt.plot(output_data,color ='red')
plt.title('Recoved Data ')

plt.subplot(3,1,1)
plt.plot(incomplete_data,color = 'blue')
plt.title('Incompleted Data')
plt.tight_layout()
plt.show()
plt.savefig('对比图.png')

# import matplotlib.pyplot as plt 

# # 1. 确保拿到的是纯粹的一维 NumPy 数组
# # 我们直接从 batch 里拿数据，并强行扁平化
# orig = label_batch.detach().cpu().numpy().flatten()
# recon = output_batch.detach().cpu().numpy().flatten()
# incomp = input_batch.detach().cpu().numpy().flatten()

# # 调试打印：如果这三个数是 0，说明数据压根没传进来
# print(f"检查数据长度: 原始={len(orig)}, 修复={len(recon)}, 残缺={len(incomp)}")

# plt.figure(figsize=(12, 10))

# # 第一张：残缺输入 (Input)
# plt.subplot(3, 1, 1)
# plt.plot(incomp, color='blue', linewidth=1)
# plt.title('1. Incomplete Data (Input)')
# plt.grid(True)

# # 第二张：修复结果 (Output)
# plt.subplot(3, 1, 2)
# plt.plot(recon, color='red', linewidth=1)
# plt.title('2. Recovered Data (Model Output)')
#                                               

# # 第三张：标准答案 (Label)
# plt.subplot(3, 1, 3)
# plt.plot(orig, color='green', linewidth=1)
# plt.title('3. Original Data (Target)')
# plt.grid(True)

# plt.tight_layout()
# plt.savefig('对比图_v2.png')
# plt.show() # 如果你在支持 GUI 的环境下能直接弹窗