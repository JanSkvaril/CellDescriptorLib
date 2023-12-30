"""
This script calculates descriptors from CellDescriptorsLib
for given image and its mask (or two directories)
and exports all the data into a JSON file.

Usage:
- Run the script with Python to perform the calculation and export.
- Provide necessary inputs via command-line arguments.
    - (in case of no inputs, script uses testdata)

Author: Denisa Rudincová
Date: 16. 11. 2023
"""

# import argparse
import os
import sys
import json
import pickle
import skimage.io as io
import skimage.measure as measure
import numpy as np

from DescriptorLib import descriptor_provider
from typing import Any, Dict

"""
TODO:

- create options and flags
- analyze whole directories
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
    res = descriptor_provider.ComputeForAll(image, mask)
    for key, value in res.items():
        results[key] = value
    return True


def dict_elements_to_string(dictionary: Dict[str, Any], show_types: bool)\
        -> Dict[str, Any]:
    """
    Transforms all values that are not JSON seriazable into a tuple of orignal
    type and string value.

    Returns JSON serializable dictionary
    """
    new_dictionary: Dict[str, Any] = {}

    for key, value in dictionary.items():
        if isinstance(value, Dict):
            new_dictionary[key] = dict_elements_to_string(dictionary[key],
                                                          show_types)

        elif not type(value) in [str, int, float, bool, list]:
            x = str(value)
            new_dictionary[key] = (str(type(value)).split("\'")[1], x)\
                if show_types else x
        else:
            new_dictionary[key] = value

    return new_dictionary


def export_results_json(results: Dict[str, Any],
                        show_types=False,
                        filename="output") -> None:
    """
    Exports results into a JSON file.
    """

    string_results = dict_elements_to_string(results, show_types)
    content = json.dumps(string_results, indent=4)

    with open(filename+".json", "w") as file:
        file.write(content)

    return


def export_results_pickle(results: Dict[str, Any],
                          filename: str = "output") -> None:
    """
    Exports results into a pickle file.
    """

    with open(filename+".pkl", "wb") as file:
        pickle.dump(results, file)

    return True


def analyze_image(img_path, mask_path):
    image = io.imread(img_path)
    mask = io.imread(mask_path)

    regions = measure.regionprops(mask)

    results: Dict[str, Any] = {}

    for region in regions:
        id = region.label

        results[id] = {}
        results[id]["bbox"] = region.bbox

        min_row, min_col, max_row, max_col = region.bbox
        region_img = np.copy(image[min_row:max_row + 1, min_col:max_col + 1])
        region_mask = np.copy(mask[min_row:max_row + 1, min_col:max_col + 1])
        region_mask = (region_mask == id).astype(int)

        calculate_descriptors(region_img, region_mask, results[id])

    export_results_json(results)
    export_results_pickle(results)
    return


def main() -> bool:
    img_path = "./tests/testdata/cell_img.tif"\
        if len(sys.argv) < 3 else sys.argv[1]
    mask_path = "./tests/testdata/cell_mask.tif"\
        if len(sys.argv) < 3 else sys.argv[2]

    if (not os.path.exists(img_path)):
        print(f"The path {img_path} does not exists")
        return False

    if (not os.path.exists(mask_path)):
        print(f"The path {mask_path} does not exists")
        return False

    analyze_image(img_path, mask_path)
    return


if __name__ == '__main__':
    main()
