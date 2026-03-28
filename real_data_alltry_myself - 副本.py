import numpy as np 
import torch
from torch.utils.data import Dataset,DataLoader,random_split
from obspy import read
from skimage.metrics import structural_similarity as ssim 
data = read('real_test.sgy')
data = np.stack([tr.data for tr in data])

def calculate_ssim(pred,gt):
    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()[0,0]
    if torch.is_tensor(gt):
        gt = gt.detach().cpu().numpy()[0,0]
    gt = (gt-np.min(gt))/(np.max(gt)-np.min(gt)+1e-8)
    pred = (pred-np.min(pred))/(np.max(pred)-np.min(pred)+1e-8)
    return ssim(pred,gt,data_range=1.0,gaussian_weights=True,use_sample_converiance =False,win_size=7)
    

def calculate_snr(pred,gt):
    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()
    if torch.is_tensor(gt):
        gt =gt.detach().cpu().numpy()
    mse = np.mean((pred-gt)**2)
    if mse == 0:
        return float('inf')
    signal_power =  np.mean(pred**2)
    snr  = 10*np.log10(signal_power / mse)
    return snr
def random_mask(shape,missing_rate):
    mask = (torch.rand(shape)>missing_rate).float()
    return mask 
def block_mask(shape,block_size):
    C,H,W = shape
    mask = torch.ones(shape)
    start = torch.randint(0,W-block_size,(1,)).item()
    mask[:,:,start:start+block_size] = 0
    return mask 
# class SeismicDataset(Dataset):
#     def __init__(self,data,P,S,experiment,missing_rate=None,block_size=None):
#         super().__init__()
        
        
#         self.data = (data-data.mean()) / data.std()
#         self.H,self.W  = self.data.shape
#         self.s = S
#         self.p = P
#         process_list = []
#         self.process_list = process_list
#         self.experiment = experiment
#         self.missing_rate =missing_rate
#         self.block_size = block_size
#         for i in range(0,self.H-self.p,self.s):
#             for j in range(0,self.W - self.p,self.s):
#                 process_list.append((i,j))
#     def __len__(self):
#         return len(self.process_list)
#     def __getitem__(self, idx):
#         i, j = self.process_list[idx]
#         clean_patch = self.data[i:i+self.p,j:j+self.p]
#         y = torch.from_numpy(clean_patch.astype(np.float32)).unsqueeze(0)
#         if self.experiment == 'A':
#             mask =block_mask(y.shape,self.block_size)
#         else:
#             mask = random_mask(y.shape,self.missing_rate)
#         x = y *mask
#         return x, y ,mask
import torch 
from  obspy  import read
import numpy as np
from torch.utils.data import DataLoader,Dataset,random_split
from scipy.signal import hilbert
def random_mask(shape,missing_rate):
    
    mask = (torch.rand(shape)>missing_rate).float()
    return mask 
def block_mask(shpae,block_size ):
    h,w = shpae
    mask = torch.ones(shpae)
    start = torch.randint(0,w-block_size,(1,)).item()
    mask[:,start:start+block_size] = 0
    return mask


class RandomShift(object):
    def __init__(self,max_shift):
        self.max_shift = max_shift
    def __call__(self,sample):
        shift =np.random.randint (-self.max_shift,self.max_shift)
        if shift == 0:
            return sample
        if not torch.is_tensor(sample):
            sample= torch.from_numpy(sample.astype(np.float32))            
        
        return  torch.roll(sample,shifts=shift,dims=0)
class PhaseRotation(object):
    def __init__(self,max_rotation):
        self.max_rotation= max_rotation
    def __call__(self, sample):
        theta = np.deg2rad(np.random.uniform(-self.max_rotation,self.max_rotation))
        analystic_data = hilbert(sample,axis=0 )
        data = np.real(analystic_data  *  np.exp(-1j *theta))
        return data







class Compose2(object):
    def __init__(self,transformer,mix_p,alpha):
        self.transformer= transformer
        self.data = None
        self.mix_p = mix_p
        self.alpha = alpha
    def __call__(self,sample):
        if self.transformer:
            for t in self.transformer:
                sample = t(sample)
        if np.random.random()<self.mix_p:
            sample =sample
            h,w = self.data.shape
            ph,pw = sample.shape
            start_h=(np.random.randint(0,h-ph))
            start_w=(np.random.randint(0,w-pw))
            p_b = self.data[start_h:start_h+ph,start_w:start_w+pw]
            lam = np.random.beta(self.alpha,self.alpha)
            if self.transformer:
                for t in self.transformer:
                    p_b =t(p_b)
            if not torch.is_tensor(p_b):
                    p_b = torch.from_numpy(p_b.astype(np.float32))
            if not torch.is_tensor(sample):
                    sample = torch.from_numpy(sample.astype(np.float32))
            return  lam *(p_b)+(1-lam)*sample
        else:
            return sample
        
class SeismicDataset(Dataset):
    def __init__(self,transform,S,P,experiment):
        super().__init__()
        r  =read('real_test.sgy')
        data = np.stack([tr.data for tr in r]).astype(np.float32)
        self.transform = transform
        self.data = data
        self.p = P
        self.s = S
        T,X = data.shape
        process_list =[]
        self.process_list = process_list
        
        self.transform.data =self.data 
        self.experiment =experiment
        for i in range(0,T-self.p,self.s):
            for j  in range(0,X-self.p,self.s):
                process_list.append((i,j))
        
    def  __len__(self):
        return len(self.process_list)
    def __getitem__(self, idx):
        i,j = self.process_list[idx]
        clean_patch =self.data[i:i+self.p,j:j+self.p]
        if not torch.is_tensor(clean_patch):
            clean_patch = torch.from_numpy(clean_patch.astype(np.float32))
        if self.experiment =='A':
            mask = block_mask(clean_patch.shape,16)
        else:
            mask = random_mask(clean_patch.shape,0.6)
        dirty_patch = mask *clean_patch
        sample = dirty_patch
        
        if self.transform:
            sample = self.transform(sample)
        if  not torch.is_tensor(sample):
            sample = torch.from_numpy(sample.astype(np.float32))
        return sample.unsqueeze(0),clean_patch.unsqueeze(0),mask
my_transformer = [RandomShift(10),PhaseRotation(20)]
my_compose = Compose2(my_transformer,0.6,0.4)
seismic_dataset = SeismicDataset(my_compose,32,64,'A')




trian_size  =int(0.8 *len(seismic_dataset))
test_size=  len(seismic_dataset)-trian_size
train_dataset,test_dataset = random_split(seismic_dataset,[trian_size,test_size])
train_loader = DataLoader(train_dataset,batch_size=32)
test_loader = DataLoader(test_dataset,batch_size=32)



import time
def pocs_interplotation(missing_data,gt,mask,iter_num,):
    temp = missing_data.copy()
    for i  in range(iter_num):
        
       
        fk_data = np.fft.fftshift(np.fft.fft2(temp))
        current_shold = np.max(np.abs(fk_data)) *(1-(1+i)/iter_num)
        fk_data[np.abs(fk_data)<current_shold] = 0
        xt_data = np.real(np.fft.ifft2(np.fft.ifftshift(fk_data)))
        temp = (1-mask) *xt_data +mask*missing_data
        mse = np.mean((temp-gt)**2)
        signal_power = np.mean(gt **2)
        snr = 10*np.log10((signal_power/mse)+1e-8)
         
    return temp,snr

import torch.nn as nn 
import torch.optim as optim 
class SeismicCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.CNN = nn.Sequential(nn.Conv2d(in_channels=1,out_channels=16,stride=2,kernel_size=3,padding=1),
                                 nn.ReLU(),nn.Conv2d(in_channels=16,out_channels=32,stride=2,kernel_size=3,padding=1),
                                 nn.ReLU(),nn.ConvTranspose2d(in_channels=32,out_channels=16,stride=2,kernel_size=4,padding=1),
                                 nn.ReLU(),nn.ConvTranspose2d(in_channels=16,out_channels=1,stride=2,kernel_size=4,padding=1))
    def forward(self,x):
         x = self.CNN(x)
         return x
class    SpecialLoss(nn.Module):
    def __init__(self,lambda_fk):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lambda_fk = lambda_fk
        
    def  forward(self,pred,gt,mask):
        mse_loss = self.mse(pred,gt)
        hole_mask = 1-mask
        fk_inputs = torch.log10(torch.abs(torch.fft.fft2(pred))+1e-8)
        fk_gt = torch.log10(torch.abs(torch.fft.fft2(gt))+1e-8)
        fk_loss = self.mse(fk_inputs,fk_gt)
        s1_loss =((pred-gt)**2).sum()/((hole_mask).sum()+1e-8)
        return  mse_loss +s1_loss+self.lambda_fk*fk_loss
    
device = torch.device('cuda')
model = SeismicCNN().to(device)
creterion = SpecialLoss(lambda_fk=2)
optimizer = optim.Adam(model.parameters(),lr= 0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer,step_size =30,gamma =0.1)
model.train()
import matplotlib.pyplot as plt
def comparison_plot(input,output,gt,subtitle):
    if torch.is_tensor(output):
        output =output.detach().cpu().numpy()[0,0]
    if torch.is_tensor(gt):
        gt = gt.detach().cpu().numpy()[0,0]
    if torch.is_tensor(input):
        input = input.detach().cpu().numpy()[0,0]
    fig,axes  = plt.subplots(3,1,figsize =(12,8))
    titles =['Raw','Recovered','Origin']
    data_list =[]
    data_list.append(input)
    data_list.append(output)
    data_list.append(gt)
    
    for ax,title,data in zip(axes,titles,data_list):
        ax.imshow(data,cmap = 'seismic',aspect= 'auto')
        ax.set_title(title)
        
    plt.tight_layout()
    plt.savefig(subtitle,dpi =300)
    plt.show()
loss_list =[]
snr_list =[]

for epoch in range(100):
    total_train_loss = 0 
    total_train_ssim = 0
    for batch_idx,(inputs,labels,mask) in enumerate(train_loader):
        train_inputs =inputs.to(device)
        train_labels = labels.to(device)
        mask = mask.to(device)
        optimizer.zero_grad()
        train_outputs = model(train_inputs)
        train_loss = creterion(train_outputs,train_labels,mask)
        train_loss.backward()
        optimizer.step()
        total_train_loss +=train_loss.item()
        train_final_outputs = (1-mask) *train_outputs +mask*train_labels
        train_ssim = calculate_ssim(train_final_outputs,train_labels)
        total_train_ssim +=train_ssim
    Avg_ssim= total_train_ssim/len(train_loader)    
    snr =calculate_snr(train_outputs,train_labels)
    Avg_loss= total_train_loss/len(train_loader)
    loss_list.append(Avg_loss)
    snr_list.append(snr)
    scheduler.step()

    if (epoch+1) %10 == 0 :
            
            print(f'Epoch : {epoch+1}/100,Avg Loss:{Avg_loss},SSIM :{Avg_ssim}')
final_trian_outputs = (1-mask)*train_outputs + mask*train_labels
snr_train_cnn = 10*torch.log10(torch.mean((train_labels)**2)/torch.mean((train_labels-final_trian_outputs)**2)+1e-8)
print(f'训练SNR为：{snr_train_cnn}')
comparison_plot(train_inputs,train_outputs,labels,'Train2')
fig,ax =plt.subplots(2,1,figsize = (12,8))
ax[0].plot(loss_list,color = 'green')
ax[0].set_title('LOSS')
ax[1].plot(snr_list,color ='blue')
ax[1].set_title('SNR')

plt.grid(True, alpha=0.3)
plt.savefig('LOSS和SNR 趋势图')
plt.show()




model.eval()
with torch.no_grad():
    total_test_loss = 0
    for batch_idx,(inputs,labels,mask) in enumerate(test_loader):
        test_inputs =inputs.to(device)
        test_labels =labels.to(device)
        mask = mask.to(device)
        test_outputs = model(test_inputs)
        test_loss = creterion(test_outputs,test_labels,mask)
        total_test_loss +=test_loss.item()
        
    print(f'测试完成,误差为:{total_test_loss/len(test_loader)}')
    comparison_plot(test_inputs,test_outputs,test_labels,'Test2')
    final_outputs = (1-mask) *test_outputs +mask*test_labels
    snr_test_cnn = 10*torch.log10(torch.mean((test_labels)**2)/torch.mean((test_labels-final_outputs)**2)+1e-8)
print(f'SNR—CNN 为：{snr_test_cnn}')






for batch_idx,(inputs,labels,mask) in enumerate(test_loader):
    pocs_inputs = inputs.detach().cpu().numpy()[0,0]
    pocs_labels = labels.detach().cpu().numpy()[0,0]
    mask = mask.detach().cpu().numpy()[0,0]
    pocs_outputs,snr = pocs_interplotation(pocs_inputs,pocs_labels,mask,iter_num=50)
    pocs_ssim = calculate_ssim(pocs_outputs,pocs_labels)
    print(f'POCS 迭代完成，SNR为：{snr},SSIM:{pocs_ssim}')
start_time =time.time()
comparison_plot(pocs_inputs,pocs_outputs,pocs_labels,'POCS迭代2')
end_time =time.time()
print(f'POCS迭代的时间为{end_time - start_time}')


def plot_residual(gt,cnn_recovered,pocs_recovered,save_name):
    if torch.is_tensor(gt):
        gt = gt.detach().cpu().numpy()[0,0]
    if torch.is_tensor(cnn_recovered):
        cnn_recovered = cnn_recovered.detach().cpu().numpy()[0,0]
    if torch.is_tensor(pocs_recovered):
        pocs_recovered=pocs_recovered.detach().cpu().numpy()[0,0]
    cnn_residual =cnn_recovered-gt
    pocs_residual = pocs_recovered -gt
    fig,axes = plt.subplots(2,2,figsize=(18,6))
    data_list =[]
    data_list.append(cnn_recovered), data_list.append(pocs_recovered), data_list.append(cnn_residual), data_list.append(pocs_residual)
    titles =['CNN(Recovered)','POCS(Recovered)','CNN(RESIDUAL)','POCS(RESIDUAL)']
    for ax ,title, data in zip(axes.flatten(),titles,data_list):
        im = ax.imshow(data,aspect='auto',cmap= 'seismic')
        ax.set_title(title)
        fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(save_name)
    plt.show()
plot_residual(test_labels,test_outputs,pocs_outputs,'RESIDUAL')



def calculate_CSLR(outputs,gt,mask):
    if torch.is_tensor(outputs):
        outputs = outputs.detach().cpu().numpy()
    if torch.is_tensor(gt):
        gt = gt.detach().cpu().numpy()
    if torch.is_tensor(mask):
        mask =mask.detach().cpu().numpy()
    residual = outputs -gt
    hole_mask = 1-mask
    res_energy = np.sum((residual*hole_mask)**2)
    all_energy = np.sum(gt**2)
    cslr= res_energy /(all_energy+1e-8)
    return cslr.item()

cnn_cslr= calculate_CSLR(test_outputs,test_labels,mask)
pocs_cslr =calculate_CSLR(pocs_outputs,pocs_labels,mask)
print(f'两次实验的CSLR: CNN:{cnn_cslr},POCS:{pocs_cslr}')   