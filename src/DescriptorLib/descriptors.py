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


def MaskDecriptors(mask: np.array) -> dict:
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


def Histogram(image: np.array, mask: np.array, bins: int | None = None)\
        -> np.array:
    """
        Computes the histogram of the image within the mask.
        - image: 2D numpy array
        - mask: 2D numpy array, binary mask
        - bins: number of bins for the histogram, if none, max value of image
          is used
        Returns a 1D numpy array with the histogram.
    """
    if bins is None:
        bins = np.max(image)
    return np.histogram(image, weights=mask, bins=bins)[0]


def HistogramDescriptors(histogram: np.array) -> dict:
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
        Polar coordinates are (1, [0,pi/4,pi/2,3pi/4]). Therefore, the matrix
        is 255x255x1x4.
        - image: 2D numpy array, with values ranging from 0 to 255
        - mask: 2D numpy array, binary mask
        Returns co-ocurrence matrix
    """
    src = np.copy(image)
    src[mask == 0] = 0
    matrix = graycomatrix(src, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4],
                          levels=256, normed=True)
    matrix = matrix[1:, 1:, :, :]  # throw away zeros
    return matrix


def GlcmFeatures(matrix: np.array, mean: bool = False) -> dict:
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


def Granulometry(image: np.array, mask: np.array, max_size=10, step=1)\
     -> np.array:
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


def PowerSpectrum(image: np.array, mask: np.array, size: int | None = None)\
     -> np.array:
    """
        Calculates the power spectrum of the image within the mask.
        - image: 2D numpy array
        - mask: 2D numpy array, binary mask.
        returns 2D image containg powerscpetrum
    """
    src = np.copy(image)
    src[mask == 0] = 0  # TODO: check if this is correct
    wimage = src * window('hann', src.shape)
    wimage = wimage - np.mean(wimage)
    freq = np.abs(np.fft.fft2(wimage))
    freq = np.fft.fftshift(freq)
    if size is not None:
        freq = resize(freq, (size, size))
    return np.square(freq)


def Autocorrelation(image: np.array, mask: np.array, size: int | None = None)\
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


def LocalBinaryPattern(image: np.array, mask: np.array):
    """
        Computes histogram of local binary patterns from image within mask.
    """
    radius = 3
    n_points = 8 * radius
    pb = local_binary_pattern(image, n_points, radius, method="uniform")

    return Histogram(pb, mask, bins=n_points+1)


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
