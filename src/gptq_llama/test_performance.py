import torch
import torch.nn as nn
import time
import quant_cuda
import os
import matplotlib.pyplot as plt
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

print('Benchmarking LLaMa-7B FC2 matvec ...')

DEV = torch.device('cuda:0')

batches = [1, 3, 5, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200]
L = 128
M = 4096
N = 11008
COUNT = 100
DTYPE = torch.half

print('testing FP16 ...')

tms_fp16 = []
for B in batches:
    
    mat = torch.randn((M, N), device=DEV, dtype=DTYPE)
    vec = torch.randn((B, M), device=DEV, dtype=DTYPE)
    mul = torch.zeros((B, N), device=DEV, dtype=DTYPE)
    
    # warm-up
    for _ in range(3):
        for _ in range(COUNT):
            torch.matmul(vec, mat, out=mul) 
            torch.cuda.synchronize()
    
    tick = time.time()
    for _ in range(COUNT):
        torch.matmul(vec, mat, out=mul) 
        torch.cuda.synchronize()
    tm = (time.time() - tick) / COUNT

    tms_fp16.append(tm)

print('testing cuda kernels ...')

tms_float_2bit = []
tms_half_2bit = []
tms_float_3bit = []
tms_half_3bit = []
tms_float_4bit = []
tms_half_4bit = []

for B in batches:

    DTYPE = torch.float
    mul = torch.zeros((B, N), device=DEV, dtype=DTYPE)
    scales = torch.randn(N, device=DEV, dtype=DTYPE)
    vec = torch.randn((B, M), device=DEV, dtype=DTYPE)

    # 2bit
    mat = torch.randint(-1000000000, 1000000000, (M // 256 * 16, N), device=DEV, dtype=torch.int)
    zeros = torch.randint(-1000000000, 1000000000, (1, N // 256 * 16), device=DEV, dtype=torch.int)

    # warm-up
    for _ in range(3):
        vec = vec.float()
        for _ in range(COUNT):
            quant_cuda.vecquant2matmul(vec, mat, mul, scales, zeros, M)
            torch.cuda.synchronize()
            
    vec = vec.float()
    tick = time.time()
    for _ in range(COUNT):
        quant_cuda.vecquant2matmul(vec, mat, mul, scales, zeros, M)
        torch.cuda.synchronize()
    tm = (time.time() - tick) / COUNT
    tms_float_2bit.append(tm)

    vec = vec.half()
    tick = time.time()
    for _ in range(COUNT):
        quant_cuda.vecquant2matmul_faster(vec, mat, mul, scales, zeros, M, M//2)
        torch.cuda.synchronize()
    tm = (time.time() - tick) / COUNT
    tms_half_2bit.append(tm)

    # 3bit
    mat = torch.randint(-1000000000, 1000000000, (M // 256 * 24, N), device=DEV, dtype=torch.int)
    zeros = torch.randint(-1000000000, 1000000000, (1, N // 256 * 24), device=DEV, dtype=torch.int)

    # warm-up
    for _ in range(3):
        vec = vec.float()
        for _ in range(COUNT):
            quant_cuda.vecquant3matmul(vec, mat, mul, scales, zeros, M)
            torch.cuda.synchronize()
            
    vec = vec.float()
    tick = time.time()
    for _ in range(COUNT):
        quant_cuda.vecquant3matmul(vec, mat, mul, scales, zeros, M)
        torch.cuda.synchronize()
    tm = (time.time() - tick) / COUNT
    tms_float_3bit.append(tm)

    vec = vec.half()
    tick = time.time()
    for _ in range(COUNT):
        quant_cuda.vecquant3matmul_faster(vec, mat, mul, scales, zeros, M, M//2)
        torch.cuda.synchronize()
    tm = (time.time() - tick) / COUNT
    tms_half_3bit.append(tm)

    # 4bit
    mat = torch.randint(-1000000000, 1000000000, (M // 256 * 32, N), device=DEV, dtype=torch.int)
    zeros = torch.randint(-1000000000, 1000000000, (1, N // 256 * 32), device=DEV, dtype=torch.int)

    # warm-up
    for _ in range(3):
        vec = vec.float()
        for _ in range(COUNT):
            quant_cuda.vecquant4matmul(vec, mat, mul, scales, zeros, M)
            torch.cuda.synchronize()
            
    vec = vec.float()
    tick = time.time()
    for _ in range(COUNT):
        quant_cuda.vecquant4matmul(vec, mat, mul, scales, zeros, M)
        torch.cuda.synchronize()
    tm = (time.time() - tick) / COUNT
    tms_float_4bit.append(tm)

    vec = vec.half()
    tick = time.time()
    for _ in range(COUNT):
        quant_cuda.vecquant4matmul_faster(vec, mat, mul, scales, zeros, M, M//2)
        torch.cuda.synchronize()
    tm = (time.time() - tick) / COUNT
    tms_half_4bit.append(tm)

    print('batch_num:', B, 'completed.')

plt.figure()
plt.plot(batches, tms_fp16)
plt.plot(batches, tms_float_2bit)
plt.plot(batches, tms_half_2bit)
plt.plot(batches, tms_float_3bit)
plt.plot(batches, tms_half_3bit)
plt.plot(batches, tms_float_4bit)
plt.plot(batches, tms_half_4bit)
plt.legend(['FP16 (torch.matmul)', '2bit (float)', '2bit (half)', '3bit (float)', '3bit (half)', '4bit (float)', '4bit (half)'])
plt.grid()
plt.title('Performance of matmul')
plt.ylabel('seconds / matmul')
plt.xlabel('batch_num')
plt.savefig('result.png')
plt.close()
