import numpy as np
import struct

def generate_raw_segy(filename="raw_data.sgy", nt=512, nx=128):
    # SEG-Y 采样间隔 (2ms = 2000微秒)
    dt_us = 2000 
    
    with open(filename, "wb") as f:
        # 1. 文本头 (3200字节，必须有)
        f.write(b' ' * 3200)
        
        # 2. 二进制卷头 (400字节)
        bin_head = bytearray(400)
        bin_head[16:18] = struct.pack(">H", dt_us) # 采样率
        bin_head[24:26] = struct.pack(">H", nt)    # 采样点
        bin_head[20:22] = struct.pack(">H", 5)     # 5 = IEEE Float (最稳格式)
        f.write(bin_head)
        
        # 3. 数据道 (Traces)
        for i in range(nx):
            # 道头 (240字节)
            tr_head = bytearray(240)
            tr_head[114:116] = struct.pack(">H", nt)
            tr_head[116:118] = struct.pack(">H", dt_us)
            f.write(tr_head)
            
            # 物理数据：大端序 4字节浮点 (Big-endian float32)
            # 模拟反射界面
            data = np.random.normal(0, 0.02, nt).astype('>f4') 
            for depth in [150, 280, 420]:
                idx = int(depth + 0.6 * i) # 模拟地层倾斜
                if 0 <= idx < nt:
                    data[max(0, idx-2):min(nt, idx+3)] += 1.0
            
            f.write(data.tobytes())

    print(f">>> 文件已生成: {filename}")

if __name__ == "__main__":
    generate_raw_segy()