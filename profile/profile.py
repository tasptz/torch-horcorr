import torch

from horcorr import HorCorrFunction

def profile():
    batch = 8
    height = 128
    width = 256
    n_windows = 64
    dim = 128

    device = torch.device('cuda:0')
    inp0 = torch.empty((batch, dim, height, width + n_windows - 1), dtype=torch.float32, device=device)
    inp1 = torch.empty((batch, dim, height, width), dtype=torch.float32, device=device)

    inp0.uniform_()
    inp1.uniform_()

    inp0.requires_grad_()
    inp1.requires_grad_()

    corr = HorCorrFunction.apply(inp0, inp1)
    corr.sum().backward()

    inp0.grad.data.cpu()
    inp1.grad.data.cpu()

    torch.cuda.profiler.cudart().cudaProfilerStop()

if __name__ == '__main__':
    profile()