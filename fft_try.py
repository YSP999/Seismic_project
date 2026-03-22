from obspy import read 
import numpy as np 
import torch 
rection  = read('real_test.sgy')
data = np.stack([tr.data for tr in rection ])
fk_data = np.fft.fft2(data)
fk_shifted = np.fft.fftshift(fk_data)

fk_log1 = np.abs(np.log10(20*fk_shifted+1e-6))
fk_log2 = np.abs(np.log10(20*fk_data+ 1e-6))
import matplotlib.pyplot  as plt 
plt.figure(figsize=(12,8))

plt.subplot(1,2,1)
im = plt.imshow(fk_log1,cmap ='gray',aspect='auto')
plt.colorbar(label='Amtitude')
plt.title('Figure 1')
plt.grid(True)


plt.subplot(1,2,2)
im = plt.imshow(fk_log2,cmap ='gray',aspect='auto')
plt.colorbar(label='Amtitude')
plt.title('Figure 2')
plt.grid(True)



plt.show()
plt.savefig("灰度对比图生成.png")