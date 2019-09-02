import torch

from . import horcorr_cuda
from .horcorr_cuda import *

class HorCorrFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left, right):
        assert left.shape[1] == horcorr_cuda.channels()
        assert (left.shape[0] * left.shape[2]) % horcorr_cuda.block_size() == 0
        assert left.shape[3] - right.shape[3] == horcorr_cuda.n_windows() - 1
        ctx.save_for_backward(left, right)
        return horcorr_cuda.forward(left, right)

    @staticmethod
    def backward(ctx, grad):
        left, right = ctx.saved_tensors
        grad_left = grad_right = None
        if ctx.needs_input_grad[0]:
            grad_left = horcorr_cuda.backward_left(right, grad)
        if ctx.needs_input_grad[1]:
            grad_right = horcorr_cuda.backward_right(left, grad)
        return grad_left, grad_right