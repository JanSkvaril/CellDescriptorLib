import numpy as np


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
