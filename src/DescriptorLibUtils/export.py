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

import argparse
import os
import json
import pickle
import sys
import skimage.io as io
import skimage.measure as measure
import numpy as np

from DescriptorLib import descriptor_provider
from DescriptorLibUtils import ExportOptions
from skimage import img_as_ubyte
from typing import Any, Dict, Tuple

from tqdm import tqdm

"""
TODO:

- create options and flags
- optimalize passing/saving cells
- parallelize the process
- add more todo
"""

# so the full numpy array is printed
np.set_printoptions(threshold=np.inf)

# you can edit these "flags" according
# to your expectations :)

EXPORT_JSON = 1
EXPORT_PICKLE = 1
EXPORT_REGION_IMGS = 1
MODE_3D = 0
REMOVE_BACKGROUND = 0
CROP_OFFSET = 0

EXPORT_OPTIONS = ExportOptions()


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


def dict_elements_to_string(dictionary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transforms all values that are not JSON seriazable into a tuple of orignal
    type and string value.

    Returns JSON serializable dictionary
    """
    new_dictionary: Dict[str, Any] = {}

    for key, value in dictionary.items():
        if isinstance(value, Dict):
            new_dictionary[key] = dict_elements_to_string(dictionary[key])

        elif not type(value) in [str, int, float, bool, list]:
            new_dictionary[key] = str(value)
        else:
            new_dictionary[key] = value

    return new_dictionary


def export_to_json(results: Dict[str, Any],
                   filename: str,
                   directory: str) -> None:
    """
    Exports results into a JSON file.
    """

    string_results = dict_elements_to_string(results)
    content = json.dumps(string_results, indent=4)

    with open(f"{directory}/{filename}.json", "w") as file:
        file.write(content)

    return


def export_to_pickle(results: Dict[str, Any],
                     filename: str,
                     directory: str) -> None:
    """
    Exports results into a pickle file.
    """

    with open(f"{directory}/{filename}.pkl", "wb") as file:
        pickle.dump(results, file)

    return


def export_region_images(results: Dict[str, Any],
                         image: Any,
                         mask: Any,
                         frame_directory: str) -> None:

    for id in results.keys():
        cell_directory = f"{frame_directory}/{id}"
        if (not os.path.exists(cell_directory)):
            os.mkdir(cell_directory)

        min_row, min_col, max_row, max_col = results[id]["bbox"]
        region_mask = np.copy(mask[min_row:max_row + 1, min_col:max_col + 1])
        region_mask = img_as_ubyte((region_mask == id).astype(bool))
        region_img = np.copy(image[min_row:max_row + 1, min_col:max_col + 1])

        if EXPORT_OPTIONS.get_remove_background():
            region_img[region_mask == 0] = 0

        io.imsave(f"{cell_directory}/image.tiff", region_img)
        io.imsave(f"{cell_directory}/mask.tiff", region_mask)
    return


def export_results(results: Dict[str, Any],
                   image: Any,
                   mask: Any,
                   filename: str = "output",
                   set_directory: str = "output") -> None:

    # set directory
    if (not os.path.exists(set_directory)):
        os.mkdir(set_directory)

    # frame directory
    frame_directory = f"{set_directory}/{filename}"
    if (not os.path.exists(frame_directory)):
        os.mkdir(frame_directory)

    if EXPORT_OPTIONS.get_export_json():
        export_to_json(results, filename, frame_directory)

    if EXPORT_OPTIONS.get_export_pickle():
        export_to_pickle(results, filename, frame_directory)

    if EXPORT_OPTIONS.get_export_region_imgs():
        export_region_images(results, image, mask, frame_directory)

    return


def analyze_image(img_path: str, mask_path: str) -> None:
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

    filename = os.path.splitext(os.path.basename(img_path))[0]

    export_results(results, image, mask, filename)
    return


def analyze_directory(img_path: str, mask_path: str) -> None:
    images = sorted(os.listdir(img_path))
    masks = sorted(os.listdir(mask_path))

    for i in tqdm(range(len(images))):
        analyze_image(os.path.join(img_path, images[i]),
                      os.path.join(mask_path, masks[i]))
    return


def process_arguments() -> Tuple[str, str]:
    """
    Processes arguments from the command line.

    Returns the image and mask path.
    """
    parser = argparse.ArgumentParser(description="Process arguments.")

    # if no arguments are provided, the script uses testdata
    if len(sys.argv) == 1:
        return "./tests/testdata/images", "./tests/testdata/masks"

    # required arguments
    parser.add_argument("image_path", type=str, help="Path to the image.")
    parser.add_argument("mask_path", type=str, help="Path to the mask.")

    # optional arguments
    parser.add_argument("-2D", action="store_true", help="Enable 2D mode.")
    parser.add_argument("-3D", action="store_true", help="Enable 3D mode.")

    parser.add_argument("-removebg",
                        action="store_true",
                        help="Remove the background."
                             "Background will be set to 0."
                             "Default is keeping the background.")

    parser.add_argument("-cropoffset",
                        type=int,
                        help="Set the crop offset."
                             "0 means no offset. Default is 0.")

    parser.add_argument("-pickle",
                        action="store_true",
                        help="Enable export to pickle.")

    parser.add_argument("-json",
                        action="store_true",
                        help="Enable export to json.")

    args = parser.parse_args()

    if args._3D:
        EXPORT_OPTIONS.set_mode_3d(1)

    if args.removebg:
        EXPORT_OPTIONS.set_remove_background(1)

    if args.cropoffset:
        EXPORT_OPTIONS.set_crop_offset(args.cropoffset)

    if args.pickle:
        EXPORT_OPTIONS.set_export_pickle(1)

    if args.json:
        EXPORT_OPTIONS.set_export_json(1)

    return args.image_path, args.mask_path


def main() -> bool:
    img_path, mask_path = process_arguments()

    if (not os.path.exists(img_path)):
        print(f"The path {img_path} does not exists")
        return False

    if (not os.path.exists(mask_path)):
        print(f"The path {mask_path} does not exists")
        return False

    if (os.path.isdir(img_path) and os.path.isdir(mask_path)):
        analyze_directory(img_path, mask_path)

    elif (os.path.isfile(img_path) and os.path.isfile(mask_path)):
        analyze_image(img_path, mask_path)

    else:
        print("Both paths must lead to files"
              " or both must lead to directories.")
        return False

    return True


if __name__ == '__main__':
    main()
