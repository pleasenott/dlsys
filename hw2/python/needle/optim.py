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
        for i, param in enumerate(self.params):
            if i not in self.u:
                # NOTE: intialized with zeros
                self.u[i] = 0
            # NOTE: param.grad maybe None
            if param.grad is None:
                continue
            grad_data = ndl.Tensor(param.grad.numpy(), dtype='float32').data \
                        + self.weight_decay * param.data
            self.u[i] = self.momentum * self.u[i] \
                        + (1 - self.momentum) * grad_data
            param.data = param.data - self.u[i] * self.lr

        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        # NOTE: this function is not used in this assignment




        # raise NotImplementedError()

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
        for i, param in enumerate(self.params):
            if i not in self.m:
                self.m[i] = ndl.init.zeros(*param.shape)
                self.v[i] = ndl.init.zeros(*param.shape)

            # NOTE: param.grad maybe None
            if param.grad is None:
                continue
            grad_data = ndl.Tensor(param.grad.numpy(), dtype='float32').data \
                        + param.data * self.weight_decay
            self.m[i] = self.beta1 * self.m[i] \
                        + (1 - self.beta1) * grad_data
            self.v[i] = self.beta2 * self.v[i] \
                        + (1 - self.beta2) * grad_data ** 2
            # NOTE: bias correction
            u_hat = (self.m[i]) / (1 - self.beta1 ** self.t)
            v_hat = (self.v[i]) / (1 - self.beta2 ** self.t)
            param.data = param.data - self.lr * u_hat / (v_hat ** 0.5 + self.eps)

        #raise NotImplementedError()
        ### END YOUR SOLUTION
