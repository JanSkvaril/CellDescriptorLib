import numpy as np
from scipy.stats import moment, gmean, skew, kurtosis, entropy
from skimage.measure import moments_central, moments_hu, moments, \
    perimeter  # , regionprops
from skimage.feature import graycomatrix, graycoprops
from skimage.morphology import closing, opening, disk, convex_hull_image
from skimage.filters import window
from skimage.transform import resize
from skimage.filters import gabor
from skimage.feature import local_binary_pattern

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
            - image: 2D numpy array
            - mask: 2D numpy array, binary mask.
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
            - mask: 2D numpy array, binary mask
            Returns a dictionary with the following descriptors:
            - area
            - perimeter
            - convex perimeter
            - major axis
            - minor axis
            - bbox_size
            - elongation
            - compactness
            - circularity
            - convexity
    """

    def Eval(self, image: np.array, mask: np.array):

        result = dict()
        width = mask.shape[0]
        height = mask.shape[1]

        result["area"] = np.count_nonzero(mask)
        result["perimeter"] = perimeter(mask)
        result["convex_perimeter"] = perimeter(convex_hull_image(mask))
        result["major_axis"] = 0
        result["minor_axis"] = 0
        result["bbox_size"] = width * height
        result["elongation"] = width / height

        x = 4 * np.pi * result["area"]
        result["compactness"] = x / result["perimeter"] ** 2
        result["circularity"] = x / result["convex_perimeter"] ** 2
        result["convexity"] = result["convex_perimeter"] / result["perimeter"]

        return result

    def GetName(self) -> str:
        return "Mask descriptors"

    def GetType(self) -> DescriptorType:
        return DescriptorType.DICT_SCALAR


class StdDev(DescriptorBase):
    """
        Calculates the standard deviation of the image within the mask.
        - image: 2D numpy array
        - mask: 2D numpy array, binary mask
        Returns
    """

    def Eval(self, image: np.array, mask: np.array):
        src = np.float64(np.copy(image))
        src[mask == 0] = np.nan
        return np.nanstd(src)

    def GetName(self) -> str:
        return "StdDev"

    def GetType(self) -> DescriptorType:
        return DescriptorType.SCALAR


class Mean(DescriptorBase):
    """
        Calculates the mean of the image within the mask.
        - image: 2D numpy array
        - mask: 2D numpy array, binary mask
        Returns
    """

    def Eval(self, image: np.array, mask: np.array):
        src = np.float64(np.copy(image))
        src[mask == 0] = np.nan
        return np.nanmean(src)

    def GetName(self) -> str:
        return "Mean"

    def GetType(self) -> DescriptorType:
        return DescriptorType.SCALAR


class Histogram(DescriptorBase):
    """
        Computes the histogram of the image within the mask.
        - image: 2D numpy array
        - mask: 2D numpy array, binary mask
        - bins: number of bins for the histogram, if none, max value of image
          is used
        Returns a 1D numpy array with the histogram.
    """

    def __init__(self, bins: int | None = None):
        self.bins = bins

    def Eval(self, image: np.array, mask: np.array):

        return self.Histogram(image, mask, bins=self.bins)

    def Histogram(self, image: np.array, mask: np.array, bins: int | None = None)\
            -> np.array:
        if bins is None:
            bins = np.max(image)

        return np.histogram(image, weights=mask, bins=10)[0]

    def GetName(self) -> str:
        return "Histogram"

    def GetType(self) -> DescriptorType:
        return DescriptorType.VECTOR


class HistogramDescriptors(DescriptorBase):
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

    def Eval(self, image: np.array, mask: np.array):
        hist = Histogram()
        return self.HistogramDescriptors(hist.Eval(image, mask))

    def GetName(self) -> str:
        return "Histogram descriptors"

    def GetType(self) -> DescriptorType:
        return DescriptorType.DICT_SCALAR

    def HistogramDescriptors(self, histogram: np.array) -> dict:
        result = dict()

        dist = histogram / np.sum(histogram)
        result["mean"] = np.mean(histogram)
        result["std"] = np.std(dist)
        result["var"] = np.var(dist)
        result["median"] = np.median(histogram)
        result["max"] = np.max(histogram)
        result["min"] = np.min(histogram)
        result["argmax"] = np.argmax(histogram)
        result["moment3"] = moment(dist, moment=3)
        result["geometric_mean"] = gmean(histogram)
        result["skewness"] = skew(dist)
        result["kurtosis"] = kurtosis(dist)
        result["entropy"] = entropy(dist)
        result["energy"] = np.sum(np.square(dist))
        result["smoothness"] = 1 - 1/(1 + result["std"]**2)

        return result


class Moments(DescriptorBase):
    """
        Calculates the central moments of the image within the mask.
        - image: 2D numpy array
        - mask: 2D numpy array, binary mask
        Returns
    """

    def Eval(self, image: np.array, mask: np.array):
        src = np.copy(image)
        src[mask == 0] = 0
        return moments(src)

    def GetName(self) -> str:
        return "Moments"

    def GetType(self) -> DescriptorType:
        return DescriptorType.VECTOR


class MomentsCentral(DescriptorBase):
    """
        Calculates the central moments of the image within the mask.
        - image: 2D numpy array
        - mask: 2D numpy array, binary mask
        Returns
    """

    def Eval(self, image: np.array, mask: np.array):
        src = np.copy(image)
        src[mask == 0] = 0
        return moments_central(src)

    def GetName(self) -> str:
        return "Moments central"

    def GetType(self) -> DescriptorType:
        return DescriptorType.VECTOR


class MomentsHu(DescriptorBase):
    """
        Calculates the normalized moments of the image within the mask.
        - image: 2D numpy array
        - mask: 2D numpy array, binary mask
        Returns
    """

    def Eval(self, image: np.array, mask: np.array):
        src = np.copy(image)
        src[mask == 0] = 0
        return moments_hu(src)

    def GetName(self) -> str:
        return "Moments Hu"

    def GetType(self) -> DescriptorType:
        return DescriptorType.VECTOR


class GlcmFeatures(DescriptorBase):
    """
        Computes the descriptors of the co-ocurrence matrix.
        - matrix: glcm matrix
        - mean: if true, returns the mean of the descriptors
        Returns a dictionary with the following descriptors:
        - contrast
        - dissimilarity
        - homogeneity
        - ASM
        - energy
        - correlation
        - entropy
        - max - maximum propability
    """

    def Eval(self, image: np.array, mask: np.array):
        glcm = self.Glcm(image, mask)
        return self.GlcmFeatures(glcm)

    def GetName(self) -> str:
        return "Glcm features"

    def GetType(self) -> DescriptorType:
        return DescriptorType.DICT_SCALAR

    def Glcm(self, image: np.array, mask: np.array) -> np.array:
        """
            Creates gray level co-ocurrence matrix from values in mask.
            Polar coordinates are (1, [0,pi/4,pi/2,3pi/4]). Therefore, the matrix
            is 255x255x1x4.
            - image: 2D numpy array, with values ranging from 0 to 255
            - mask: 2D numpy array, binary mask
            Returns co-ocurrence matrix
        """
        src = np.copy(image).astype(np.uint8)
        src[mask == 0] = 0
        matrix = graycomatrix(src, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4],
                              levels=256, normed=True)
        matrix = matrix[1:, 1:, :, :]  # throw away zeros
        return matrix

    def GlcmFeatures(self, matrix: np.array, mean: bool = False) -> dict:
        """
            Computes the descriptors of the co-ocurrence matrix.
            - matrix: glcm matrix
            - mean: if true, returns the mean of the descriptors
            Returns a dictionary with the following descriptors:
            - contrast
            - dissimilarity
            - homogeneity
            - ASM
            - energy
            - correlation
            - entropy
            - max - maximum propability
        """
        result = dict()
        result["contrast"] = graycoprops(matrix, prop="contrast")
        result["dissimilarity"] = graycoprops(matrix, prop="dissimilarity")
        result["homogeneity"] = graycoprops(matrix, prop="homogeneity")
        result["ASM"] = graycoprops(matrix, prop="ASM")
        result["energy"] = graycoprops(matrix, prop="energy")
        result["correlation"] = graycoprops(matrix, prop="correlation")
        result["entropy"] = entropy(matrix, axis=(0, 1))
        result["max"] = np.max(matrix, axis=(0, 1))

        if mean:
            for k in result:
                result[k] = np.mean(result[k])
        return result


class Granulometry(DescriptorBase):
    """
        Creates granulometric curve from values in mask.
        - image: 2D numpy array
        - mask: 2D numpy array, binary mask
        - max_size: maximum size of the structuring element, every elemnt up
          to this size will be used
        - step: step between sizes

        Returns granulometric curve, that shows size distributions of objects
        in the image
    """

    def Eval(self, image: np.array, mask: np.array):
        return self.Granulometry(image, mask)

    def GetName(self) -> str:
        return "Granulometry"

    def GetType(self) -> DescriptorType:
        return DescriptorType.VECTOR

    def Granulometry(self, image: np.array, mask: np.array, max_size=10, step=1)\
            -> np.array:
        sums = []

        for size in range(max_size, 0, step*-1):
            se = disk(size)
            im = closing(image, se)
            im[mask == 0] = 0
            sums.append(np.sum(im))

        for size in range(0, max_size, step):
            se = disk(size)
            im = opening(image, se)
            im[mask == 0] = 0
            sums.append(np.sum(im))
        curve = np.array(sums)
        curve = -np.abs(np.diff(curve))
        curve = curve - np.min(curve)
        curve = curve / np.max(curve)
        return curve


class PowerSpectrum(DescriptorBase):
    """
        Calculates the power spectrum of the image within the mask.
        - image: 2D numpy array
        - mask: 2D numpy array, binary mask.
        returns 2D image containg powerscpetrum
    """

    def Eval(self, image: np.array, mask: np.array):
        return self.PowerSpectrum(image, mask)

    def GetName(self) -> str:
        return "Power spectrum"

    def GetType(self) -> DescriptorType:
        return DescriptorType.MATRIX

    def PowerSpectrum(self, image: np.array, mask: np.array) -> np.array:
        src = np.copy(image)
        src[mask == 0] = 0  # TODO: check if this is correct
        wimage = src * window('hann', src.shape)
        wimage = wimage - np.mean(wimage)
        freq = np.abs(np.fft.fft2(wimage))
        freq = np.fft.fftshift(freq)
        return np.square(freq)


class Autocorrelation(DescriptorBase):
    """
        Calculates the autocorrelation of the image within the mask.
        - image: 2D numpy array
        - mask: 2D numpy array, binary mask.
        returns 2D image containg autocorrelation
    """

    def Eval(self, image: np.array, mask: np.array):
        return self.Autocorrelation(image, mask)

    def GetName(self) -> str:
        return "Autocorrelation"

    def GetType(self) -> DescriptorType:
        return DescriptorType.MATRIX
    def Autocorrelation(self, image: np.array, mask: np.array, size: int | None = None)\
            -> np.array:
        """
            Calculates the autocorrelation of the image within the mask.
            - image: 2D numpy array
            - mask: 2D numpy array, binary mask.
        """
        src = np.copy(image)
        src[mask == 0] = 0

        dataFT = np.fft.fft(src, axis=1)
        dataAC = np.fft.ifft(dataFT * np.conjugate(dataFT), axis=1).real
        dataAC = np.fft.fftshift(dataAC, axes=1)

        if size is not None:
            dataAC = resize(dataAC, (size, size))
        return dataAC


class LocalBinaryPattern(DescriptorBase):
    """
        Computes histogram of local binary patterns from image within mask.
    """

    def Eval(self, image: np.array, mask: np.array):
        return self.LocalBinaryPattern(image, mask)

    def GetName(self) -> str:
        return "Local binary pattern"

    def GetType(self) -> DescriptorType:
        return DescriptorType.VECTOR

    def LocalBinaryPattern(self, image: np.array, mask: np.array):
        """
            Computes histogram of local binary patterns from image within mask.
        """
        radius = 3
        n_points = 8 * radius
        pb = local_binary_pattern(image, n_points, radius, method="uniform")
        hs = Histogram(n_points+1)
        return hs.Eval(pb, mask)


class GaborSpectralHistogram(DescriptorBase):
    def Eval(self, image: np.array, mask: np.array):
        gabors = GaborFilters(image, mask)
        return SpectralHistogram(gabors)

    def GetName(self) -> str:
        return "Gabor spectral histograms"

    def GetType(self) -> DescriptorType:
        return DescriptorType.SPECTAL_HISTOGRAM


class GaborEnergy(DescriptorBase):
    """
        Computes energy of each gabor filter response from image within mask.
    """

    def Eval(self, image: np.array, mask: np.array):
        gabors = GaborFilters(image, mask)
        return FeatureBankEnergy(gabors)

    def GetName(self) -> str:
        return "Gabor energy"

    def GetType(self) -> DescriptorType:
        return DescriptorType.VECTOR


def SpectralHistogram(feature_bank, bins=255):
    """
        Creates spectral histogram from feature bank.
        - feature_bank: list of 2D numpy arrays, if mask is used, features
          should be masked
        - bins: number of bins for the histogram, images are normalized
          to range [0, binds]
        returns list of histograms - spectral histogram
    """
    histograms = []
    for feature in feature_bank:
        f = feature - np.min(feature)
        f = f / np.max(f)
        f = f * bins
        histograms.append(np.histogram(f, bins=bins)[0][1:])
    return histograms


def GaborFilters(image: np.array, mask: np.array,
                 thetas: int = 10, frequencies: int = 10,
                 freq_range=(0.1, 0.9)) -> np.array:
    """
        Creates a bank of gabor filter responses from image within mask.
        - image: 2D numpy array
        - mask: 2D numpy array, binary mask.
        - thetas: number of angles, evenly spaced between 0 and 2pi
        - frequencies: number of frequencies, evenly spaced between
        freq_range[0] and freq_range[1]
        - freq_range: tuple of min and max frequency
        Returns list of images - gabor filter responses
    """
    featrue_bank = []
    for theta in np.linspace(0, np.pi*2, thetas):
        for freq in np.linspace(freq_range[0], freq_range[1], frequencies):
            res, _ = gabor(image, frequency=freq, theta=theta)
            res[mask == 0] = 0
            featrue_bank.append(res)
    return featrue_bank


def FeatureBankEnergy(feature_bank, normalize=False):
    """
        Computes energy of each feature in feature bank.
        - feature_bank: list of 2D numpy arrays, if mask is used, features
          should be masked
        - normalize: if true, energy is normalized to range [0,1]
        returns array of energies
    """
    energies = []
    for feature in feature_bank:
        energies.append(np.sum(np.square(feature)))
    energies = np.array(energies)
    if normalize:
        energies = energies - np.min(energies)
        energies = energies / np.max(energies)
    return energies
