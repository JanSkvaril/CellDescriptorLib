from descriptors import *


def GetAll() -> list:
    """
    Returns a list of all descriptors.
    """
    desc = []
    desc.append(MaskDecriptors())
    desc.append(HistogramDescriptors())
    desc.append(Moments())
    desc.append(MomentsCentral())
    desc.append(MomentsHu())
    desc.append(GlcmFeatures())
    desc.append(Granulometry())
    desc.append(PowerSpectrum())
    desc.append(Autocorrelation())
    desc.append(LocalBinaryPattern())

    desc.append(GaborEnergy())
    # desc.append(GaborSpectralHistogram())
    return desc


def ComputeForAll(image: np.array, mask: np.array) -> dict:
    """
    Computes all descriptors for the given image and mask.
    returns dictionary with descriptor name as key, value is tuple (type, result)
    """
    results = {}
    for desc in GetAll():
        results[desc.GetName()] = (desc.GetType(), desc.Eval(image, mask))
    return results
