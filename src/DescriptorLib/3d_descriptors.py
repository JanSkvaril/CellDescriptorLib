import numpy as np
#from scipy.stats import moment, gmean, skew, kurtosis, entropy
#from skimage.measure import moments_central, moments_hu, moments, \
#    perimeter, label
#from skimage.feature import graycomatrix, graycoprops
#from skimage.morphology import closing, opening, disk, convex_hull_image
#from skimage.filters import window
#from skimage.transform import resize
#from skimage.filters import gabor
#from skimage.feature import local_binary_pattern
#from skimage import io
from math import pi
from scipy.spatial import ConvexHull
from porespy.metrics import regionprops_3D

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


class MaskDecriptors(DescriptorBase):
    """
            Calculates descriptors of the given mask.
            - mask: 3D numpy array, binary mask
            Returns a dictionary with the following descriptors:
            - surface area
            - volume
            - bbox_volume
            - major axis length
            - minor axis length
            - compactness
            - sphericity
            - elongation
            - convexity
    """

    def Eval(self, image: np.array, mask: np.array):
        result = dict()
        width, height, depth = mask.shape

        props = regionprops_3D(mask)[0]

        result["surface_area"] = props.surface_area
        result["volume"] = np.count_nonzero(mask)
        
        result["bbox_volume"] = width * height * depth
        result["major_axis"] = props.axis_major_length
        result["minor_axis"] = props.axis_minor_length

        result["compactness"] = (36 * pi * (result["volume"] ** 2)) / (result["surface_area"] ** 3)
        result["sphericity"] = result["compactness"] ** (1/3)

        foreground_points = np.argwhere(mask)
        convex_hull = ConvexHull(foreground_points)
        convex_hull_volume = convex_hull.volume
        result["convexity"] = result["volume"] / convex_hull_volume

        result["elongation"] = result["major_axis"] / result["minor_axis"]

        return result

    def GetName(self) -> str:
        return "Mask descriptors"

    def GetType(self) -> DescriptorType:
        return DescriptorType.DICT_SCALAR
