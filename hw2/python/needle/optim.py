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
        for w in self.params:
          # print (w, w.grad, w.data)
          if w.grad == None:
            continue
          if w not in self.u:
            self.u[w] = 0
          gradient_decayed = w.grad.data + w.data * self.weight_decay
          self.u[w] = self.momentum * self.u[w] + (1 - self.momentum) * gradient_decayed
          w.data = w.data + (-self.lr) * self.u[w]
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
        for w in self.params:
          if w.grad == None:
            continue
          if w not in self.m:
            self.m[w] = 0
            self.v[w] = 0
          gradient_decayed = w.grad + w.data * self.weight_decay
          self.m[w] = self.beta1 * self.m[w] + (1 - self.beta1) * gradient_decayed
          self.v[w] = self.beta2 * self.v[w] + (1 - self.beta2) * (gradient_decayed ** 2)
          m_hat = self.m[w] / (1 - self.beta1 ** self.t)
          v_hat = self.v[w] / (1 - self.beta2 ** self.t)
          w.data = w.data + (-self.lr) * m_hat / ((v_hat ** 0.5) + self.eps)
        ### END YOUR SOLUTION
