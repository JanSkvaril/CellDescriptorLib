import numpy as np
from .descriptors2d import MaskDecriptors, HistogramDescriptors, \
    Moments, MomentsCentral, MomentsHu, GlcmFeatures, Granulometry, \
    PowerSpectrum, Autocorrelation, LocalBinaryPattern, GaborEnergy
# from .descriptors3d import MaskDescriptors3D


def GetAll(d: int = 2) -> list:
    """
    Returns a list of all descriptors.
    """
    desc = []

    # 2D descriptors
    if d == 2:
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

    # if d == 3:
    #     desc.append(MaskDescriptors3D())
    return desc


def ComputeForAll(image: np.array, mask: np.array, d: int = 2) -> dict:
    """
    Computes all descriptors for the given image and mask.
    returns dictionary with descriptor name as key, value is tuple
    (type, result)
    """
    results = {}
    for desc in GetAll():
        results[desc.GetName()] = (desc.GetType(), desc.Eval(image, mask))
    return results
