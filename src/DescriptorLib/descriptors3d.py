import numpy as np
from math import pi, sqrt
from scipy.spatial import ConvexHull
from porespy.metrics import regionprops_3D
from skimage.morphology import ball
from scipy.ndimage import binary_opening
from skimage.transform import resize

from .descriptor import DescriptorBase, DescriptorType


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
        - histogram: 1D numpy array

        Returns a dictionary with the following descriptors:
        - mean
        - standard deviation
        - variance
        - median
        - max
        - min
        - argmax
        - geometric_mean
        - skewness
        - kurtosis
        - entropy
        - energy
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

    def getKthMoment(self, k: int, values: np.array, histogram: np.array,
                     mean: float, count: int) -> float:

        return np.sum(((values - mean) ** k) * histogram) / count

    def HistogramDescriptors3D(self,
                               histogram: np.array,
                               bin_edges: np.array) -> dict:
        result = dict()

        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        total_count = np.sum(histogram)

        result["mean"] = np.sum(bin_centers * histogram) / total_count

        moment2 = self.getKthMoment(2, bin_centers, histogram,
                                    result["mean"], total_count)

        moment3 = self.getKthMoment(3, bin_centers, histogram,
                                    result["mean"], total_count)

        moment4 = self.getKthMoment(4, bin_centers, histogram,
                                    result["mean"], total_count)

        result["var"] = moment2
        result["std"] = sqrt(result["var"])
        result["max"] = bin_edges[len(bin_edges) - 1]
        result["min"] = bin_edges[0]

        cumulative_sum = np.cumsum(histogram)
        median_index = np.argmax(cumulative_sum >= np.sum(histogram) / 2)

        bin_width = bin_edges[1] - bin_edges[0]
        median_bin_left = bin_edges[median_index]
        median_bin_count = histogram[median_index]

        total_counts_before_median = 0
        if median_index > 0:
            total_counts_before_median = cumulative_sum[median_index - 1]

        result["median"] = median_bin_left + ((total_count / 2
                                               - total_counts_before_median)
                                              / median_bin_count) * bin_width

        result["argmax"] = np.argmax(histogram)

        total_contribution = np.prod(bin_centers * histogram)
        result["gmean"] = total_contribution ** (1 / total_count)

        result["skewness"] = moment3 / (result["std"] ** 3)
        result["kurtosis"] = (moment4 / (result["var"] ** 2)) - 3

        prob_dist = histogram / total_count
        prob_dist = prob_dist[prob_dist != 0]
        result["entropy"] = -np.sum(prob_dist * np.log2(prob_dist))

        result["energy"] = np.sum(histogram ** 2)

        return result


class Granulometry3D(DescriptorBase):
    """
        Creates granulometric curve from values in the image within the mask.

        - image: 3D numpy array
        - mask: 3D numpy array, binary mask
        - max_radius: maximum radius of the structuring element,
        - step: step between sizes of structuring element

        Returns granulometric curve, that shows size distributions of objects
        in the image.
    """

    def Eval(self, image: np.array, mask: np.array):
        return self.Granulometry3D(image, mask)

    def GetName(self) -> str:
        return "Granulometry"

    def GetType(self) -> DescriptorType:
        return DescriptorType.VECTOR

    def Granulometry3D(self,
                       image: np.array,
                       mask: np.array,
                       max_radius=10,
                       step=1) -> np.array:

        curve = []
        init_volume = np.sum(mask)

        for radius in range(0, max_radius, step):
            st_element = ball(radius)
            opening = binary_opening(image, structure=st_element)
            opening[mask == 0] = 0

            opening_volume = np.sum(opening)
            curve.append(opening_volume - init_volume)
            init_volume = opening_volume

        curve = np.array(curve)
        curve = curve - np.min(curve)
        curve = curve / np.max(curve)

        return curve


class PowerSpectrum3D(DescriptorBase):
    """
        Calculates the power spectrum of the image within the mask.

        - image: 3D numpy array
        - mask: 3D numpy array, binary mask.

        returns 3D image containg powerscpetrum
    """

    def Eval(self, image: np.array, mask: np.array) -> np.array:
        img = np.copy(image)
        img[mask == 0] = 0

        fft_image = np.fft.fftn(img)
        power_image = np.abs(fft_image) ** 2

        return np.fft.fftshift(power_image)

    def GetName(self) -> str:
        return "Power spectrum"

    def GetType(self) -> DescriptorType:
        return DescriptorType.MATRIX


class Autocorrelation3D(DescriptorBase):
    """
        Calculates the autocorrelation of the image within the mask.

        - image: 3D numpy array
        - mask: 3D numpy array, binary mask.

        returns 3D image containg autocorrelation
    """

    def Eval(self, image: np.array, mask: np.array):
        return self.Autocorrelation3D(image, mask)

    def GetName(self) -> str:
        return "Autocorrelation"

    def GetType(self) -> DescriptorType:
        return DescriptorType.MATRIX

    def Autocorrelation3D(self,
                          image: np.array,
                          mask: np.array,
                          size: int | None = None) -> np.array:
        src = np.copy(image)
        src[mask == 0] = 0

        var = np.var(src)
        data = src - np.mean(src)

        power_img = np.abs(np.fft.fftn(data)) ** 2
        autocorr_img = np.fft.ifftn(power_img).real / var / np.prod(src.shape)

        if size is not None:
            autocorr_img = resize(autocorr_img, (size, size, size))

        return autocorr_img
