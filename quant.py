import numpy as np
import torch
import torch.nn as nn


def quantize(x, scale, zero, maxq):
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)

class Quantizer(nn.Module):

    def __init__(self, shape=1):
        super(Quantizer, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('zero', torch.zeros(shape))

    def configure(
            self,
            bits, perchannel=False, sym=True, 
            mse=False, norm=2.4, grid=100, maxshrink=.8
        ):
        self.maxq = torch.tensor(2 ** bits - 1)
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink 

    def find_params(self, x, weight=False):
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            if weight:
                x = x.flatten(1)
            else:
                if len(shape) == 4:
                    x = x.permute([1, 0, 2, 3])
                    x = x.flatten(1)
                if len(shape) == 3:
                    x = x.reshape((-1, shape[-1])).t()
                if len(shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        self.scale = (xmax - xmin) / self.maxq
        if self.sym:
            self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
        else:
            self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float('inf'), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid 
                xmin1 = p * xmin
                xmax1 = p * xmax
                scale1 = (xmax1 - xmin1) / self.maxq
                zero1 = torch.round(-xmin1 / scale1) if not self.sym else self.zero
                q = quantize(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)
                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
        if not self.perchannel:
            if weight:
                tmp = shape[0]
            else:
                tmp = shape[1] if len(shape) != 3 else shape[2]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        if weight:
            shape = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(shape)
            self.zero = self.zero.reshape(shape)
            return
        if len(shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
            self.zero = self.zero.reshape((1, -1, 1, 1))
        if len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero = self.zero.reshape((1, 1, -1)) 
        if len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero = self.zero.unsqueeze(0)

    def quantize(self, x):
        if self.ready():
            return quantize(x, self.scale, self.zero, self.maxq)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)


try:
    import quant_cuda
except:
    print('CUDA extension not installed.')

# Assumes layer is perfectly divisible into 256 * 256 blocks
class QuantLinear(nn.Module): 
    def __init__(self, bits, infeatures, outfeatures):
        super().__init__()
        if bits not in [2,3,4,8]:
            raise NotImplementedError("Only 2,3,4,8 bits are supported.")
        self.bits = bits
        self.register_buffer('zeros', torch.zeros((outfeatures, 1)))
        self.register_buffer('scales', torch.zeros((outfeatures, 1)))
        self.register_buffer('bias', torch.zeros(outfeatures))
        self.register_buffer(
            'qweight', torch.zeros((infeatures // 256 * (bits * 8), outfeatures), dtype=torch.int)
        )
        if self.bits == 4:
            self.register_buffer('wf1', torch.tensor([0, 4, 8, 12, 16, 20, 24, 28], dtype=torch.int32).unsqueeze(0).unsqueeze(2), persistent=False)
            self.register_buffer('wf2', torch.tensor([0, 4, 8, 12, 16, 20, 24, 28], dtype=torch.int32).unsqueeze(0).unsqueeze(0), persistent=False)
        elif self.bits == 3:
            self.register_buffer('wf1', torch.tensor([
                [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 0],
                [0, 1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31],
                [0, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 0],
            ], dtype=torch.int32).reshape(1,3,12,1), persistent=False)
            self.register_buffer('wf2', torch.tensor([
                [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 0],
                [0, 1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31],
                [0, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 0],
            ], dtype=torch.int32).reshape(1,1,3,12), persistent=False)


    def pack(self, linear, scales, zeros):
        self.zeros = zeros * scales
        self.scales = scales.clone()
        if linear.bias is not None:
            self.bias = linear.bias.clone()  

        intweight = torch.round((linear.weight.data + self.zeros) / self.scales).to(torch.int)
        intweight = intweight.t().contiguous()
        intweight = intweight.numpy().astype(np.uint32)
        qweight = np.zeros(
            (intweight.shape[0] // 256 * (self.bits * 8), intweight.shape[1]), dtype=np.uint32
        )
        i = 0
        row = 0
        while row < qweight.shape[0]:
            if self.bits in [2,4,8]:
                for j in range(i, i + (32//self.bits)):
                    qweight[row] |= intweight[j] << (self.bits * (j - i))
                i += 32//self.bits
                row += 1
            elif self.bits == 3:
                for j in range(i, i + 10):
                    qweight[row] |= intweight[j] << (3 * (j - i))
                i += 10
                qweight[row] |= intweight[i] << 30
                row += 1
                qweight[row] |= (intweight[i] >> 2) & 1
                i += 1
                for j in range(i, i + 10):
                    qweight[row] |= intweight[j] << (3 * (j - i) + 1)
                i += 10
                qweight[row] |= intweight[i] << 31
                row += 1
                qweight[row] |= (intweight[i] >> 1) & 0x3
                i += 1
                for j in range(i, i + 10):
                    qweight[row] |= intweight[j] << (3 * (j - i) + 2)
                i += 10
                row += 1
            else:
                raise NotImplementedError("Only 2,3,4,8 bits are supported.")
                
        qweight = qweight.astype(np.int32)
        self.qweight = torch.from_numpy(qweight) 

    def forward(self, x):
        outshape = list(x.shape)
        x = x.reshape(-1, x.shape[-1])
        y = self.bias.clone().repeat(x.shape[0],1)
        if x.shape[-2] >= 128 and self.bias == None:
            # Fall back to PyTorch matmul
            if self.bits == 4 :
                # Unpack 4bit weights
                weight = torch.bitwise_right_shift(torch.unsqueeze(self.qweight, 1).expand(-1, 8, -1), self.wf1).to(torch.int8)
                torch.bitwise_and(weight, 0x0000000F, out=weight)
                weight = weight.reshape(-1, self.groupsize, weight.shape[2])

                zeros = torch.bitwise_right_shift(torch.unsqueeze(self.qzeros, 2).expand(-1, -1, 8), self.wf2).to(torch.int8)
                torch.bitwise_and(zeros, 0x0000000F, out=zeros)
                zeros = zeros + 1
                zeros = zeros.reshape(-1, zeros.shape[1] * zeros.shape[2])

                scales = self.scales

                weights = (scales * (weight - zeros))
                weights = weights.reshape(weights.shape[0] * weight.shape[1], weights.shape[2])
                x = torch.matmul(x, weights.to(x.dtype))
                return x
            if self.bits == 3:
                # Unpack 3bit weights

                weight = self.qweight.reshape(self.qweight.shape[0]//3, 3, 1, self.qweight.shape[1]).expand(-1, -1, 12, -1)
                weight = (weight >> self.wf1)&0x7
                weight[:,0,10] = (weight[:,0,10]&0x3) | ((weight[:,1,0] << 2)&0x4)
                weight[:,1,11] = (weight[:,1,11]&0x1) | ((weight[:,2,0] << 1)&0x6)
                weight = weight & 0x7
                weight = torch.cat([weight[:,0,:11], weight[:,1,1:12], weight[:,2,1:11]], dim=1)
                weight = weight.reshape(-1, self.groupsize, weight.shape[2])

                zeros = self.qzeros.reshape(self.qzeros.shape[0], self.qzeros.shape[1]//3, 3, 1).expand(-1, -1, -1, 12)
                zeros = (zeros >> self.wf2)
                zeros[:,:,0,10] = (zeros[:,:,0,10]&0x3) | ((zeros[:,:,1,0] << 2)&0x4)
                zeros[:,:,1,11] = (zeros[:,:,1,11]&0x1) | ((zeros[:,:,2,0] << 1)&0x6)
                zeros = zeros & 0x7
                zeros = torch.cat([zeros[:,:,0,:11], zeros[:,:,1,1:12], zeros[:,:,2,1:11]], dim=2)
                zeros = zeros.reshape(-1, 1, zeros.shape[1] * zeros.shape[2])
                zeros = zeros + 1

                scales = self.scales
                scales = scales.reshape(-1, 1, scales.shape[-1])

                weights = (scales * (weight - zeros))
                weights = weights.reshape(weights.shape[0] * weight.shape[1], weights.shape[2])
                x = torch.matmul(x, weights.to(x.dtype))

                return x

        outshape[-1] = self.bias.numel()
        dtype = x.dtype
        x = x.float()
        if self.bits == 2:
            quant_cuda.vecquant2matmul(x, self.qweight, y, self.scales, self.zeros)
        elif self.bits == 3:
            quant_cuda.vecquant3matmul(x, self.qweight, y, self.scales, self.zeros)
        elif self.bits == 4:
            quant_cuda.vecquant4matmul(x, self.qweight, y, self.scales, self.zeros)
        elif self.bits == 8:
            quant_cuda.vecquant8matmul(x, self.qweight, y, self.scales, self.zeros)
        else:
            raise NotImplementedError("Only 2,3,4,8 bits are supported.")
        y = y.to(dtype)
        return y.reshape(outshape)

def make_quant(module, names, bits, name=''):
    if isinstance(module, QuantLinear):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + '.' + attr if name != '' else attr
        if name1 in names:
            setattr(
                module, attr, QuantLinear(bits, tmp.in_features, tmp.out_features)
            )
    for name1, child in module.named_children():
        make_quant(child, names, bits, name + '.' + name1 if name != '' else name1)
