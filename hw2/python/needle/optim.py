"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for param_id ,param in enumerate(self.params):
            u_next = self.momentum * self.u.get(param_id, 0) + (1 - self.momentum) * (param.grad + self.weight_decay * param.data)
            self.u[param_id] = ndl.Tensor(u_next, dtype=param.dtype)
            param.data -= self.lr * self.u[param_id]
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for param_id, param in enumerate(self.params):
            if param.grad is None:
                continue
            m_next = self.beta1 * self.m.get(param_id, 0) + (1 - self.beta1) * (param.grad + self.weight_decay * param.data)
            self.m[param_id] = ndl.Tensor(m_next, dtype=param.dtype)
            v_next = self.beta2 * self.v.get(param_id, 0) + (1 - self.beta2) * (param.grad + self.weight_decay * param.data) ** 2
            self.v[param_id] = ndl.Tensor(v_next, dtype=param.dtype)
            m_hat = self.m[param_id] / (1 - self.beta1 ** self.t)
            v_hat = self.v[param_id] / (1 - self.beta2 ** self.t)
            param.data -= self.lr * m_hat / (v_hat ** 0.5 + self.eps)
        ### END YOUR SOLUTION
