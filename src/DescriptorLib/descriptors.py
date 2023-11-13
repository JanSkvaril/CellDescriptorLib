import numpy as np
from scipy.stats import moment, gmean,skew,kurtosis, entropy
from skimage.measure import moments_central, moments_hu, moments,moments_normalized

def StdDev(image: np.array, mask: np.array) -> float:
    """
        Calculates the standard deviation of the image within the mask.
        - image: 2D numpy array
        - mask: 2D numpy array, binary mask
        Returns 
        """

    src = np.float64(np.copy(image))
    src[mask == 0] = np.nan
    return np.nanstd(src)


def Mean(image: np.array, mask: np.array) -> float:
    """
        Calculates the mean of the image within the mask.
        - image: 2D numpy array
        - mask: 2D numpy array, binary mask
        Returns 
        """

    src = np.float64(np.copy(image))
    src[mask == 0] = np.nan
    return np.nanmean(src)



def Histogram(image : np.array, mask : np.array, bins:int|None = None) -> np.array:
    """
        Computes the histogram of the image within the mask.
        - image: 2D numpy array
        - mask: 2D numpy array, binary mask
        - bins: number of bins for the histogram, if none, max value of image is used
        Returns a 1D numpy array with the histogram.
    """
    if bins is None:
        bins = np.max(image)
    return np.histogram(image, weights=mask,bins=bins)[0]


def HistogramDescriptors(histogram : np.array) -> dict:
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
    result = dict()

    dist = histogram/ np.sum(histogram)
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
    result["smoothness"] = 1 - 1/(1 +  result["std"]**2)

    
    return result


def Moments(image: np.array, mask: np.array) -> np.array:
    """
        Calculates the central moments of the image within the mask.
        - image: 2D numpy array
        - mask: 2D numpy array, binary mask
        Returns 
        """
    src = np.copy(image)
    src[mask == 0] = 0
    return moments(src)

def MomentsCentral(image: np.array, mask: np.array) -> np.array:
    """
        Calculates the central moments of the image within the mask.
        - image: 2D numpy array
        - mask: 2D numpy array, binary mask
        Returns 
        """
    src = np.copy(image)
    src[mask == 0] = 0
    return moments_central(src)



def MomentsHu(image: np.array, mask: np.array) -> np.array:
    """
        Calculates the normalized moments of the image within the mask.
        - image: 2D numpy array
        - mask: 2D numpy array, binary mask
        Returns 
        """
    src = np.copy(image)
    src[mask == 0] = 0
    return moments_hu(src)


