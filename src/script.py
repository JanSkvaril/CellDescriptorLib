"""
This script calculates descriptors from CellDescriptorsLib
and exports all the data into a JSON file.

Usage:
- Run the script with Python to perform the calculation and export.
- Provide necessary inputs via command-line arguments or user prompts.
  - (in case of no inputs, script uses testdata)

Author: Denisa Rudincová
Date: 16. 11. 2023
"""

import sys
import json
# import pickle
import skimage.io as io
import skimage.measure as measure
import numpy as np

from DescriptorLib import MaskDecriptors, Histogram, HistogramDescriptors, \
    Moments, MomentsCentral, MomentsHu
from typing import Any, Dict

"""
TODO:

- validate inputs
- create options and flags
- calculate more descriptors
- pickle and json?
"""


def calculate_descriptors(image: np.ndarray,
                          mask: Any,
                          results: Dict[str, Any]) -> bool:
    """
    Calculates the descriptors and stores them in one dictionary.

    TODO:
        - extend to options, user can choose which descriptors
        will be calculated

    Returns
    """

    results["Mask Descriptors"] = MaskDecriptors(mask)
    histogram: Dict[str, Any] = Histogram(image, mask)
    results["Histrogram Descriptors"] = HistogramDescriptors(histogram)

    results["Moments"] = Moments(image, mask)
    results["Central Moments"] = MomentsCentral(image, mask)
    results["Hu Moments"] = MomentsHu(image, mask)

    return True


def dict_elements_to_string(dictionary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transforms all values that are not JSON seriazable in a tuple of orignal
    type and string value.

    Returns JSON serializable dictionary
    """
    new_dictionary: Dict[str, Any] = {}

    for key, value in dictionary.items():
        if isinstance(value, Dict):
            new_dictionary[key] = dict_elements_to_string(dictionary[key])
        elif not type(value) in [str, int, float, bool, list]:
            new_dictionary[key] = (str(type(value)).split("\'")[1], str(value))
        else:
            new_dictionary[key] = value

    return new_dictionary


def export_results(results: Dict[str, Any]) -> bool:
    """
    Exports results into a JSON file
    """

    string_results = dict_elements_to_string(results)
    y = json.dumps(string_results, indent=4)

    with open("output.json", "w") as file:
        file.write(y)

    # with open('numpy_array.pkl', 'wb') as file:
    #     pickle.dump(results, file)

    # with open('numpy_array.pkl', 'rb') as file:
    #     unpickled = pickle.load(file)

    return True


def main() -> bool:
    img_path = "./tests/testdata/cell_img.tif"\
        if len(sys.argv) < 3 else sys.argv[1]
    mask_path = "./tests/testdata/cell_mask.tif"\
        if len(sys.argv) < 3 else sys.argv[2]

    image = io.imread(img_path)
    mask = io.imread(mask_path)

    regions = measure.regionprops(mask)

    results: Dict[str, Any] = {}

    for id, region in enumerate(regions):
        if id == 0:
            continue
        results[id] = {}
        min_row, min_col, max_row, max_col = region.bbox
        calculate_descriptors(image[min_row:max_row + 1, min_col:max_col + 1],
                              mask[min_row:max_row + 1, min_col:max_col + 1],
                              results[id])

    export_results(results)

    return True


if __name__ == '__main__':
    main()
