from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        Z_max = array_api.max(Z, axis=1, keepdims=True)
        Z_log_sum_exp = array_api.log(array_api.sum(array_api.exp(Z - Z_max), axis=1, keepdims=True)) + Z_max
        return Z - Z_log_sum_exp
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        Z = exp(Z)
        Z_sum = summation(Z, axes=1).reshape((Z.shape[0], 1))
        grad_sum = summation(out_grad, axes=1).reshape((out_grad.shape[0], 1))
        grad = grad_sum / Z_sum
        grad = grad.broadcast_to(Z.shape)
        return out_grad - grad * Z    
        ### END YOUR SOLUTION

def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        Z_max = array_api.max(Z, axis=self.axes, keepdims=True)
        Z_max_reduce = array_api.max(Z, axis=self.axes)
        return array_api.log(array_api.sum(array_api.exp(Z - Z_max), axis=self.axes)) + Z_max_reduce
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        Z_max = Z.realize_cached_data().max(axis=self.axes, keepdims=True)
        Z = Z - Z_max
        Z_sum_exp = summation(exp(Z), axes=self.axes)
        shape = list(Z.shape)
        axes = range(len(shape)) if self.axes is None else self.axes
        for axis in axes:
            shape[axis] = 1
        return (out_grad / Z_sum_exp).reshape(shape).broadcast_to(Z.shape) * exp(Z)
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

