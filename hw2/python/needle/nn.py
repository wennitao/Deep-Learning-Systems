"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []




class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        weight = init.kaiming_uniform (in_features, out_features)
        self.weight = Parameter (weight)
        if bias:
          self.bias = Parameter (init.kaiming_uniform (out_features, 1).reshape ((1, out_features)))
        else:
          self.bias = None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.bias:
          # print (X.shape)
          # print (self.weight.shape)
          # print (X)
          return X @ self.weight + self.bias.broadcast_to ((X.shape[0], self.out_features))
        else:
          return X @ self.weight
        # return ops.matmul (X, ops.transpose (self.weight)) + ops.broadcast_to (self.bias, (self.in_features, self.out_features))
        ### END YOUR SOLUTION



class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        size = 1
        for i in range (1, len (X.shape)):
          size *= X.shape[i]
        return X.reshape ((X.shape[0], size))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu (x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # print (self.modules)
        for curModule in self.modules:
          # print (curModule)
          x = curModule.forward (x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        onehot = init.one_hot (logits.shape[1], y)
        sumVec = ops.summation (logits * onehot, 1)
        resMat = ops.logsumexp (logits, (1, )) - sumVec
        return ops.divide_scalar (ops.summation (resMat), np.float32(resMat.shape[0]))
        ### END YOUR SOLUTION



class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter (init.ones (dim))
        self.bias = Parameter (init.zeros (dim))
        self.running_mean = Tensor (init.zeros (dim))
        self.running_var = Tensor (init.ones (dim))
        ### END YOUR SOLUTION


    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
          Ex = ops.summation (x, axes = 0) / x.shape[0]
          ExShaped = Ex.reshape ((1, x.shape[1])).broadcast_to (x.shape)
          # print (Ex.shape)
          dif = x - ExShaped
          Varx = ops.summation (dif ** 2, axes = 0) / x.shape[0]
          VarxShaped = Varx.reshape ((1, x.shape[1])).broadcast_to (x.shape)
          # print (Varx.shape)
          div = (VarxShaped + self.eps) ** 0.5
          # div = div.reshape ((1, x.shape[1])).broadcast_to (x.shape)
          # print (div.shape)
          mat = dif / div
          self.running_mean.data = (Ex.data * self.momentum) + (self.running_mean.data) * (1 - self.momentum)
          self.running_var.data = (Varx.data * self.momentum) + (self.running_var.data) * (1 - self.momentum)
        else:
          Ex = self.running_mean.reshape ((1, x.shape[1])).broadcast_to (x.shape)
          Varx = self.running_var.reshape ((1, x.shape[1])).broadcast_to (x.shape)
          dif = x - Ex
          div = (Varx + self.eps) ** 0.5
          mat = dif / div
        return self.weight.reshape ((1, x.shape[1])).broadcast_to (x.shape) * mat + self.bias.reshape ((1, x.shape[1])).broadcast_to (x.shape)
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter (init.ones (dim))
        self.bias = Parameter (init.zeros (dim))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        Ex = ops.summation (x, axes = 1) / x.shape[1]
        # print ('LayerNorm1d:')
        # print ('x', x)
        # print ('Ex', Ex)
        # print (Ex.op)
        dif = x - Ex.reshape ((x.shape[0], 1)).broadcast_to (x.shape)
        # print ('dif', dif)
        Varx = ops.summation (dif ** 2, axes = 1) / x.shape[1]
        # print ('Varx', Varx)
        # print (Varx.op)
        div = (Varx + self.eps) ** 0.5
        div = div.reshape ((x.shape[0], 1)).broadcast_to (x.shape)
        # print ('div', div)
        mat = dif / div
        # print ('mat', mat)
        # print (mat.op)
        return self.weight.reshape ((1, x.shape[1])).broadcast_to (x.shape) * mat + self.bias.reshape ((1, x.shape[1])).broadcast_to (x.shape)
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training == False:
          return x
        size = 1
        for i in range (len (x.shape)):
          size *= x.shape[i]
        randMat = init.randb (size, p = 1 - self.p, dtype='float32').reshape (x.shape)
        randMat = randMat / (1 - self.p)
        return x * randMat
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn.forward (x) + x
        ### END YOUR SOLUTION



