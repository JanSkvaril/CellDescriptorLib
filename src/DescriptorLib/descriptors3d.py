import numpy as np
from math import pi, sqrt
from scipy.spatial import ConvexHull
from porespy.metrics import regionprops_3D

from descriptor import DescriptorBase, DescriptorType


class MaskDescriptors3D(DescriptorBase):
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

        result["compactness"] =\
            (36 * pi * (result["volume"] ** 2)) / (result["surface_area"] ** 3)
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


class Mean3D(DescriptorBase):
    """
        Calculates the mean of the image within the mask.
        - image: 3D numpy array
        - mask: 3D numpy array, binary mask
        Returns scalar representing mean.
    """

    def Eval(self, image: np.array, mask: np.array):
        image_copy = np.copy(image)
        image_copy[mask == 0] = 0

        return np.sum(image_copy) / np.count_nonzero(mask)

    def GetName(self) -> str:
        return "Mean"

    def GetType(self) -> DescriptorType:
        return DescriptorType.SCALAR


class StdDev3D(DescriptorBase):
    """
        Calculates the standard deviation of the image within the mask.
        - image: 3D numpy array
        - mask: 3D numpy array, binary mask
        Returns scalar representing standard deviation.
    """

    def Eval(self, image: np.array, mask: np.array):
        non_zero_points = np.copy(image)[mask != 0]

        count_points = np.count_nonzero(mask)
        mean = np.sum(non_zero_points) / count_points

        src = non_zero_points.flatten()

        return sqrt(np.sum((src - mean) ** 2) / count_points)

    def GetName(self) -> str:
        return "StdDev"

    def GetType(self) -> DescriptorType:
        return DescriptorType.SCALAR


class Histogram3D(DescriptorBase):
    """
        Computes the histogram of the image within the mask.
        - image: 3D numpy array
        - mask: 3D numpy array, binary mask
        - bins: number of bins for the histogram, if none, max value of image
          is used
        Returns a 1D numpy array with the histogram.
    """

    def __init__(self, bins: int | None = None):
        self.bins = bins

    def Eval(self, image: np.array, mask: np.array):
        return self.Histogram3D(image, mask, bins=self.bins)

    def Histogram3D(self,
                    image: np.array,
                    mask: np.array,
                    bins: int | None = None) -> np.array:
        if bins is None:
            bins = np.max(image)

        return np.histogram(image[mask != 0], bins)

    def GetName(self) -> str:
        return "Histogram"

    def GetType(self) -> DescriptorType:
        return DescriptorType.VECTOR


class HistogramDescriptors3D(DescriptorBase):
    """
        Computes the descriptors of the histogram.
        - histogram: 1D numpy array, histogram
        Returns a dictionary with the following descriptors:
        - mean
        - std
        - var
        - median
        - max
        - min
        - argmax
        - moment3
        - geometric_mean
        - skewness
        - kurtosis
        - entropy
        - energy
        - smoothness

    """
    def __init__(self, bins: int | None = None):
        self.bins = bins

    def Eval(self, image: np.array, mask: np.array):
        histogram, bin_edges = Histogram3D(self.bins).Eval(image, mask)
        return self.HistogramDescriptors3D(histogram, bin_edges)

    def GetName(self) -> str:
        return "Histogram descriptors"

    def GetType(self) -> DescriptorType:
        return DescriptorType.DICT_SCALAR

    def HistogramDescriptors3D(self,
                               histogram: np.array,
                               bin_edges: np.array) -> dict:
        result = dict()

        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        result["mean"] = np.sum(bin_centers * histogram) / np.sum(histogram)
        result["var"] = np.sum(((bin_centers - result["mean"]) ** 2) * histogram) / np.sum(histogram)
        result["std"] = sqrt(result["var"])
        result["max"] = bin_edges[len(bin_edges) - 1]
        result["min"] = bin_edges[0]

        return result
