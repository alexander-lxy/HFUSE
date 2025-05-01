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

def upsample_input():
    c = range(32, 96, 2)
    for x in c:
        yield torch.randn(1, x, 256, 100, **kwargs)

input_hist = torch.randn((512 - 32) * 10000, **kwargs)

for i in upsample_input():
    result = fusion_cuda.histc_upsample(input_hist, i)
    del result
    del i

torch.cuda.empty_cache()
