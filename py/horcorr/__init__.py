import torch

from . import horcorr_cuda
from .horcorr_cuda import *

class HorCorrFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left, right):
        assert left.shape[1] == horcorr_cuda.channels()
        assert (left.shape[0] * left.shape[2]) % horcorr_cuda.block_size() == 0
        assert left.shape[3] - right.shape[3] == horcorr_cuda.n_windows() - 1

        left = left.permute(0, 2, 3, 1).contiguous()
        right = right.permute(0, 2, 3, 1).contiguous()
        ctx.save_for_backward(left, right)
        out = torch.empty(right.shape[:3] + (horcorr_cuda.n_windows(),),
            dtype=right.dtype,
            device=right.device).contiguous()
        horcorr_cuda.forward(left, right, out)
        return out.permute(0, 3, 1, 2)

    @staticmethod
    def backward(ctx, grad):
        left, right = ctx.saved_tensors
        grad = grad.permute(0, 2, 3, 1).contiguous()
        grad_left = grad_right = None
        if ctx.needs_input_grad[0]:
            grad_left = torch.empty(left.shape,
                dtype=left.dtype,
                device=left.device).contiguous()
            horcorr_cuda.backward_left(right, grad, grad_left)
            grad_left = grad_left.permute(0, 3, 1, 2)
        if ctx.needs_input_grad[1]:
            grad_right = torch.empty(right.shape,
                dtype=right.dtype,
                device=right.device).contiguous()
            horcorr_cuda.backward_right(left, grad, grad_right)
            grad_right = grad_right.permute(0, 3, 1, 2)
        return grad_left, grad_right