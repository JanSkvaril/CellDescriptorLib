import numpy as np

from abc import ABC, abstractmethod
from enum import Enum


class DescriptorType(Enum):
    # single values
    SCALAR = 1,
    # array of values, like histogram
    VECTOR = 2,
    # matrix of values / image
    MATRIX = 3,

    # returns dictionary of scalar values
    DICT_SCALAR = 4

    # array of histograms
    SPECTAL_HISTOGRAM = 5


class DescriptorBase(ABC):
    @abstractmethod
    def Eval(self, image: np.array, mask: np.array):
        """
            Computes descriptor from image within mask.
            - image: 3D numpy array
            - mask: 3D numpy array, binary mask.
            Return type depends on type, use GetType()
        """
        pass

    @abstractmethod
    def GetName(self) -> str:
        pass

    @abstractmethod
    def GetType(self) -> DescriptorType:
        pass

    def __call__(self, image: np.array, mask: np.array):
        "Same as calling eval"
        return self.Eval(image, mask)
