import numpy as np
from scipy.stats import moment, gmean,skew,kurtosis, entropy
from skimage.measure import moments_central, moments_hu, moments,moments_normalized
from skimage.feature import graycomatrix, graycoprops
from skimage.morphology import closing, opening, disk
from skimage.filters import window
from skimage.transform import resize
from skimage.filters import correlate_sparse

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


def Glcm(image: np.array, mask: np.array) -> np.array:
    """
        Creates gray level co-ocurrence matrix from values in mask. 
        Polar coordinates are (1, [0,pi/4,pi/2,3pi/4]). Therefore, the matrix is 255x255x1x4.
        - image: 2D numpy array, with values ranging from 0 to 255
        - mask: 2D numpy array, binary mask
        Returns co-ocurrence matrix
    """
    src = np.copy(image)
    src[mask == 0] = 0
    matrix = graycomatrix(src, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, normed=True)
    matrix = matrix[1:,1:,:,:] # throw away zeros
    return matrix

def GlcmFeatures(matrix:np.array, mean :bool = False) -> dict:
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
    result["entropy"] = entropy(matrix, axis=(0,1))
    result["max"] = np.max(matrix, axis=(0,1))

    if mean:
        for k in result:
            result[k] = np.mean(result[k])
    return result

def Granulometry(image: np.array, mask: np.array, max_size = 10, step = 1) -> np.array:
    """
        Creates granulometric curve from values in mask. 
        - image: 2D numpy array
        - mask: 2D numpy array, binary mask
        - max_size: maximum size of the structuring element, every elemnt up to this size will be used
        - step: step between sizes

        Returns granulometric curve, that shows size distributions of objects in the image
    """
    sums = []

    for size in range(max_size,0, step*-1):
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
    curve = curve /  np.max(curve)
    return curve
    


def PowerSpectrum(image: np.array, mask: np.array, size:int|None = None) -> np.array:
    """
        Calculates the power spectrum of the image within the mask.
        - image: 2D numpy array
        - mask: 2D numpy array, binary mask.
        returns 2D image containg powerscpetrum
        """
    src = np.copy(image)
    src[mask == 0] = 0 # TODO: check if this is correct
    wimage = src * window('hann', src.shape)
    wimage = wimage - np.mean(wimage)
    freq = np.abs(np.fft.fft2(wimage))
    freq = np.fft.fftshift(freq)
    if size is not None:
        freq = resize(freq, (size,size))
    return np.square(freq)


def Autocorrelation(image: np.array, mask: np.array, size:int|None = None) -> np.array:
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
        dataAC = resize(dataAC, (size,size))
    return dataAC



