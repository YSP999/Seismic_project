import numpy as np 
import torch 
from torch.utils.data import Dataset
from obspy import read 
rection = read('real_test.sgy')
data =  np.stack([tr for tr in rection ])
data_max = np.max(data)
data_norm = data/data_max


class SeismicDataset(Dataset):
    def __init__(self,data_list,patch_size=128):
        super().__init__()
        self.data = data_list
        self.patch_size = patch_size
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        trace =self.data[idx]
        start =np.random.randint(0,len(trace)-self.patch_size+1)
        patch =trace[start:start+self.patch_size]
        mask = np.random.choice([0,1],patch.shape,[0.7,0.3])
        incomplete_patch = mask*patch
        x = torch.from_numpy(incomplete_patch.astype(np.float32)).unsqueeze(0)
        y = torch.from_numpy(patch.astype(np.float32)).unsqueeze(0)
        return x,y 


# test_ds =SeismicDataset(data_norm,patch_size=128)
# x1,y1 = test_ds[0]
# x2,y2 = test_ds[0]
# print(f'分块形状:{x1.shape}')
# print(f"两次分块的结果一样吗?{'一样' if torch.equal(y1,y2)  else '不一样，随机完成'}")


from torch.utils.data import dataloader

