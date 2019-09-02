import torch
import numpy as np

from horcorr import HorCorrFunction

def test_horcorr():
    batch = 4
    height = 32
    width = 16
    n_windows = 64
    dim = 128
    
    def to_numpy(x):
        return x.detach().cpu().numpy()

    def get_grad(corr, inp0, inp1):
        corr.sum().backward()
        grad0 = to_numpy(inp0.grad.data)
        inp0.grad.data.zero_()
        grad1 = to_numpy(inp1.grad.data)
        inp1.grad.data.zero_()
        return to_numpy(corr), grad0, grad1

    def run(inp0, inp1, corr_scale):
        corr = HorCorrFunction.apply(inp0, inp1) * corr_scale
        return get_grad(corr, inp0, inp1)

    def run_test(inp0, inp1, corr_scale):
        u = inp0.unfold(3, inp0.shape[3] - n_windows + 1, 1)
        corr = (u * inp1[..., None, :]).sum(dim=1).permute(0, 2, 1, 3) * corr_scale
        return get_grad(corr, inp0, inp1)

    device = torch.device('cuda:0')
    inp0 = torch.empty((batch, dim, height, width + n_windows - 1), dtype=torch.float32, device=device)
    inp1 = torch.empty((batch, dim, height, width), dtype=torch.float32, device=device)
    corr_scale = torch.empty((batch, n_windows, height, width), dtype=torch.float32, device=device)

    inp0.uniform_()
    inp1.uniform_()
    corr_scale.uniform_()

    inp0 *= 10.
    inp1 *= 10.

    inp0.requires_grad_()
    inp1.requires_grad_()

    corr, grad0, grad1 = run(inp0, inp1, corr_scale)
    corr_test, grad0_test, grad1_test = run_test(inp0, inp1, corr_scale)

    assert np.allclose(corr, corr_test)
    assert np.allclose(grad0, grad0_test)
    assert np.allclose(grad1, grad1_test)

if __name__ == '__main__':
    test_horcorr()