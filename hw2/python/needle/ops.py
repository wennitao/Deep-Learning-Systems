"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
import numpy
import needle as ndl

# NOTE: we will numpy as the array_api
# to backup our computations, this line will change in later homeworks
import numpy as array_api


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple(*[out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return (out_grad[0] + out_grad[1], )


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad, )


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar, )


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.power (a, self.scalar)
        # return a.power (self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        pre = node.inputs[0]
        return (out_grad * self.scalar * array_api.power (pre, self.scalar - 1), )
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        return out_grad / rhs, -out_grad * lhs / (rhs * rhs)
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (out_grad / self.scalar, )
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.axes != None:
          return array_api.swapaxes (a, self.axes[0], self.axes[1])
        else:
          dim = len (a.shape)
          return array_api.swapaxes (a, dim - 1, dim - 2)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (out_grad.transpose(axes=self.axes), )
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # print("<Reshape compute>:", a.shape, self.shape)
        # print ('in Reshape class compute')
        # print (self, a)
        toShape = []
        for i in range (len (self.shape)):
          if self.shape[i] == 0:
            break
          toShape.append (self.shape[i])
        if isinstance (a, ndl.Tensor):
          return array_api.reshape (a.numpy(), tuple (toShape))
        return array_api.reshape (a, tuple (toShape))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # print("<Reshape gradient>:", out_grad.shape, node.inputs[0].shape)
        # print ('in Reshape class gradient')
        # print (self)
        # print (node, node.inputs)
        pre = node.inputs[0]
        # print ('reshape gradient calling reshape ', out_grad.shape, pre.shape)
        return (out_grad.reshape (pre.shape), )
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        # print("<BroadcastTo compute>:", a.shape, self.shape)
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # print("<BroadcastTo gradient>:", out_grad.shape, node.inputs[0].shape)
        pre = node.inputs[0]
        if len (out_grad.shape) == len (pre.shape):
          axes = []
          for i in range (len (out_grad.shape)):
            if out_grad.shape[i] != pre.shape[i]:
              axes.append (i)
          # print ('BroadcastTo gradient calling reshape ', out_grad.sum (axes=tuple (axes)).shape, pre.shape)
          return (out_grad.sum (axes=tuple (axes)).reshape (pre.shape), )
        else:
          axes = list (range (len (out_grad.shape)))
          idx = 0
          for i in range (len (out_grad.shape)):
            if idx < len (pre.shape) and out_grad.shape[i] == pre.shape[idx]:
              axes.remove (i)
              idx = idx + 1
          # print ('BroadcastTo gradient calling reshape ', out_grad.sum (axes=tuple (axes)).shape, pre.shape)
          return (out_grad.sum (axes=tuple (axes)).reshape (pre.shape), )
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum (a, self.axes, dtype='float32')
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        pre = node.inputs[0]
        out_shape = list(out_grad.shape)
        if self.axes == None:
          for i in range (len (pre.shape)):
            out_shape.append (1)
        elif type (self.axes) == tuple:
          for i in self.axes:
            out_shape.insert (i, 1)
        elif type (self.axes) == int:
          out_shape.append (self.axes)
        out_shape = tuple (out_shape)
        # print ('summation gradient calling reshape ', out_grad, out_shape)
        return (out_grad.reshape (out_shape).broadcast_to (pre.shape), )
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        # print (a, b)
        return array_api.matmul (a, b, dtype='float32')
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        lres = out_grad.matmul (transpose (rhs))
        rres = transpose (lhs).matmul (out_grad)
        if lres.shape != lhs.shape:
          sum_dim = tuple (range (len (lres.shape) - len (lhs.shape)))
          lres = lres.sum (axes=sum_dim)
        if rres.shape != rhs.shape:
          sum_dim = tuple (range (len (rres.shape) - len (rhs.shape)))
          rres = rres.sum (axes=sum_dim)
        return lres, rres
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (-out_grad, )
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log (a, dtype='float32')
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        pre = node.inputs[0]
        return (out_grad / pre, )
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp (a.numpy())
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        pre = node.inputs[0].numpy()
        return (out_grad * exp (pre), )
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # print (a)
        return array_api.maximum (0, a, dtype='float32')
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        pre = node.inputs[0].numpy()
        return (out_grad * (pre > 0) * 1, )
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)



class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        if self.axes == None:
          maxZ = array_api.amax (Z, self.axes)
          return array_api.log (array_api.sum (array_api.exp (Z - maxZ), self.axes), dtype='float32') + maxZ
        else:
          maxZ = array_api.amax (Z, self.axes)
          newShape = []
          for i in range (len (Z.shape)):
            if i in self.axes:
              newShape.append (1)
            else:
              newShape.append (Z.shape[i])
          subtractZ = maxZ.reshape (tuple (newShape))
          return array_api.log (array_api.sum (array_api.exp (Z - subtractZ), self.axes), dtype='float32') + maxZ
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0].numpy()
        if self.axes == None:
          maxZ = array_api.amax (Z, self.axes)
          mat = Z - maxZ
          expMat = array_api.exp (mat)
          sumMat = array_api.sum (expMat)
          mat = expMat / sumMat
          return (ndl.Tensor (mat * out_grad.numpy(), dtype='float32'), )
        else:
          newShape = []
          for i in range (len (Z.shape)):
            if i in self.axes:
              newShape.append (1)
            else:
              newShape.append (Z.shape[i])
          grad = array_api.reshape (out_grad.numpy(), tuple (newShape))
          grad = array_api.broadcast_to (grad, Z.shape)
          maxZ = array_api.amax (Z, self.axes)
          subtractZ = maxZ.reshape (tuple (newShape))
          mat = Z - subtractZ
          expMat = array_api.exp (mat)
          sumMat = array_api.sum (expMat, self.axes).reshape (tuple (newShape))
          mat = expMat / sumMat
          return (ndl.Tensor (mat * grad, dtype='float32'), )
          ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)
