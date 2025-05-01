import torch
import fusion_cuda

torch.manual_seed(42)
torch.backends.cudnn.enabled = False

device = torch.device("cuda")
dtype = torch.float32

kwargs = {
    'dtype': dtype,
    'device': device,
    'requires_grad': True
}
# 在 for i in upsample_input(): 循环中，每次迭代会生成一个形状为 (1, x, 256, 100) 的张量，
# 其中 x 取自 range(32, 96, 2) 中的数值。
def upsample_input():
    c = range(32, 96, 2)
    for x in c:
        yield torch.randn(1, x, 25600, 1000, **kwargs)

input_hist = torch.randn((512 - 32) * 1000000, **kwargs)

for i in upsample_input():
    result = fusion_cuda.histc_upsample(input_hist, i)
    del result
    del i

torch.cuda.empty_cache()
