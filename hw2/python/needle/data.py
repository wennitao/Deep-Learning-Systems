import numpy as np
from .autograd import Tensor
import gzip
import struct

from typing import Iterator, Optional, List, Sized, Union, Iterable, Any


class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        if flip_img:
          return np.flip (img, 1)
        else:
          return img
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding+1, size=2)
        ### BEGIN YOUR SOLUTION
        shift_x = -shift_x
        shift_y = -shift_y
        shiftImg = np.roll (img, (shift_x, shift_y), axis=(0, 1))
        if shift_x >= 0:
          shiftImg[:shift_x, :, :] = 0
        else:
          shiftImg[shift_x:, :, :] = 0
        if shift_y >= 0:
          shiftImg[:, :shift_y, :] = 0
        else:
          shiftImg[:, shift_y:, :] = 0
        return shiftImg 
        ### END YOUR SOLUTION


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = np.array_split(np.arange(len(dataset)), 
                                           range(batch_size, len(dataset), batch_size))

    def __iter__(self):
        ### BEGIN YOUR SOLUTION
        self.maxIndex = len (self.dataset)
        self.curIndex = 0
        self.ordering = np.arange (len (self.dataset))
        if self.shuffle:
          np.random.shuffle (self.ordering)
        ### END YOUR SOLUTION
        return self

    def __next__(self):
        ### BEGIN YOUR SOLUTION
        if (self.curIndex + 1) * self.batch_size > self.maxIndex:
          raise StopIteration
        else:
          self.curIndex += 1
          # print (self.ordering[(self.curIndex - 1) * self.batch_size : self.curIndex * self.batch_size], self.maxIndex)
          # assert self.dataset[self.ordering[(self.curIndex - 1) * self.batch_size : self.curIndex * self.batch_size]] != None
          return tuple (Tensor (x) for x in self.dataset[self.ordering[(self.curIndex - 1) * self.batch_size : self.curIndex * self.batch_size]])
          # batch_X = []
          # batch_y = []
          # for idx in range ((self.curBatchIndex - 1) * self.batch_size, self.curBatchIndex * self.batch_size):
          #   curX, cury = self.dataset[idx]
          #   batch_X.append (curX)
          #   batch_y.append (cury)
          # return (Tensor (batch_X, dtype='float32'), Tensor (batch_y, dtype='float32'))
          # return tuple ([Tensor (curArray, dtype='float32') for curArray in self.dataset[(self.curBatchIndex - 1) * self.batch_size : self.curBatchIndex * self.batch_size]])
          # returnDat = self.dataset[(self.curBatchIndex - 1) * self.batch_size : self.curBatchIndex * self.batch_size]
          # return (Tensor (returnDat[0], dtype='float32'), Tensor (returnDat[1], dtype='float32'))
        ### END YOUR SOLUTION


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms)

        with gzip.open (image_filename) as f:
          file_content = f.read()
        index = 0
        magic_number, image_num, row, col = struct.unpack_from ('>IIII', file_content, index)
        index += struct.calcsize ('>IIII')
        X_list = []
        for i in range(image_num):
          for j in range (row):
            for k in range (col):
              cur_byte = int(struct.unpack_from ('>B', file_content, index)[0])
              X_list.append (cur_byte)
              index += struct.calcsize ('>B')
        X = np.array(X_list, np.float32).reshape(image_num, row * col)
        min_value = np.min (X)
        max_value = np.max (X)
        self.X = (X - min_value) / (max_value - min_value)
        # self.X = np.reshape (self.X, (len(X), 28, 28, 1))

        with gzip.open (label_filename) as f:
          file_content = f.read()
        index = 0
        magic_number, num = struct.unpack_from ('>II', file_content, index)
        index += struct.calcsize ('>II')
        Y_list = []
        for i in range(image_num):
          cur_byte = int(struct.unpack_from ('>B', file_content, index)[0])
          Y_list.append (cur_byte)
          index += struct.calcsize ('>B')
        self.y = np.array (Y_list, np.uint8)
        # return (X, y)
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        # print (self.X[index].shape)
        if isinstance (index, int):
          curX = self.X[index].reshape ((28, 28, 1))
          curY = self.y[index]
          curX = self.apply_transforms (curX)
          return (curX.reshape (784), curY)
        elif isinstance (index, slice):
          curX = self.X[index]
          cury = self.y[index]
          for idx in range (len (curX)):
            curX[idx] = self.apply_transforms (curX[idx].reshape ((28, 28, 1))).reshape (784)
          return (curX, cury)
        else:
          curX = []
          cury = []
          for idx in index:
            curX.append (self.apply_transforms (self.X[idx].reshape ((28, 28, 1))).reshape (784))
            cury.append (self.y[idx])
          return (curX, cury)
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return len (self.X)
        ### END YOUR SOLUTION

class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])
