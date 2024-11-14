import torch
import time

# 确定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建大矩阵
size = 10000
matrix1 = torch.randn(size, size)
matrix2 = torch.randn(size, size)

# 在CPU上进行矩阵乘法
start_time = time.time()
result_cpu = torch.matmul(matrix1, matrix2)
end_time = time.time()
print(f"CPU 计算时间: {end_time - start_time} 秒")

# 将矩阵移动到GPU
matrix1 = matrix1.to(device)
matrix2 = matrix2.to(device)

# 在GPU上进行矩阵乘法
start_time = time.time()
result_gpu = torch.matmul(matrix1, matrix2)
end_time = time.time()
print(f"GPU 计算时间: {end_time - start_time} 秒")