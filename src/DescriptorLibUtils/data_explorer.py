# Author: Jan Skvaril
# Date: 2024

import glob
import os
import skimage.io as io
import pickle
import numpy as np
from typing import Tuple
from DescriptorLib import DescriptorType


class DataExplorer:
    """
    Class for exploring output of export.py script.
    It provides methods for accessing the cells by id and time frame.
    You can simply access specific descriptor for a cell,
    or get the whole timeline of a cell.

    Example usage:
    ```
    explorer = DataExplorer("output")
    explorer.GetCellDescriptorTimeline(cell_id=1, "Mask descriptors", "area")
    will output [5128, 5124, 2454, ...]
    ```
    """

    def __init__(self, path: str):
        self.path = path
        self.frames = glob.glob(f"{path}/*")

    def GetPathToFrame(self, frame: int):
        """
        Returns the path to the frame directory.
        If the frame is out of bounds, returns None
        """

        if (frame < 0 or frame >= len(self.frames)):
            return None
        return f"{self.frames[frame]}"

    def GetCellAtFrame(self, frame: int,
                       cell_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the image and mask of a cell at a given frame.
        Returns (None, None) if the cell does not exist
        """

        frame_path = self.GetPathToFrame(frame)

        image_path = f"{frame_path}/{cell_id}/image.tiff"
        mask_path = f"{frame_path}/{cell_id}/mask.tiff"
        if (not os.path.isfile(image_path) or not os.path.isfile(mask_path)):
            return None, None
        image = io.imread(image_path)
        mask = io.imread(mask_path)

        return image, mask

    def GetDescriptorsAtFrame(self, frame: int) -> dict:
        """
        Returns the descriptors of all cells at a given frame.
        Returns None if the frame does not exist
        """

        frame_path = self.GetPathToFrame(frame)
        dir_name = os.path.basename(frame_path)
        descriptors_path = f"{frame_path}/{dir_name}.pkl"

        if not os.path.isfile(descriptors_path):
            return None
        with open(descriptors_path, "rb") as file:
            descriptors = pickle.load(file)
        return descriptors

    def GetDescriptorsForCell(self, frame: int, cell_id: int) -> dict:
        """
        Returns all descriptors of a cell at a given frame.
        Returns None if the cell or frame does not exist
        """

        descriptors = self.GetDescriptorsAtFrame(frame)
        if descriptors is None:
            return None
        if cell_id not in descriptors.keys():
            return None
        return descriptors[cell_id]

    def GetCellDescriptor(self, frame: int, cell_id: int,
                          descriptor_name: str, dict_val: str = ""):
        """
        Returns specific descriptor for given cell in frame.
        If dict_val is set and the descriptor is a dictionary,
        returns the value of the key dict_val.
        If dict_val is not set, returns (descriptor_type, descriptor_value)
        """

        descriptors = self.GetDescriptorsForCell(frame, cell_id)
        if descriptors is None:
            return None
        if descriptor_name not in descriptors.keys():
            return None
        if dict_val != "" and descriptors[descriptor_name][0] == DescriptorType.DICT_SCALAR:
            return descriptors[descriptor_name][1][dict_val]
        return descriptors[descriptor_name]

    def GetAllCellsInFrame(self, frame: int):
        """
        Returns all celll ids in a given frame.
        Returns None if the frame does not exist.
        """

        descriptors = self.GetDescriptorsAtFrame(frame)
        if descriptors is None:
            return None
        return descriptors.keys()

    def GetCellTimeline(self, cell_id: int):
        """
        Returns 2 lists:
            - image timeline - list of images of the cell in each frame,
                               in order
            - mask timeline - corresponding list of masks
        If cell is not present in a frame, the frame is skipped
        """

        timeline_image = []
        timeline_mask = []
        for frame in range(len(self.frames)):
            image, mask = self.GetCellAtFrame(frame, cell_id)
            if image is None or mask is None:
                continue
            timeline_image.append(image)
            timeline_mask.append(mask)

        return timeline_image, timeline_mask

    def GetCellDescriptorTimeline(self, cell_id, descriptor_name, dict_val=""):
        """
        Returns a list of descriptor for a given cell in each frame, in order
        If dict_val is set and the descriptor is a dictionary,
        returns the value of the key dict_val.

        Example usage:
        ```
        explorer = DataExplorer("output")
        explorer.GetCellDescriptorTimeline(1, "Mask descriptors", "area")
        will output [5128, 5124, 2454, ...]
        ```
        """

        timeline = []
        for frame in range(len(self.frames)):
            descriptor = self.GetCellDescriptor(frame, cell_id,
                                                descriptor_name, dict_val)
            if descriptor is None:
                continue
            timeline.append(descriptor)
        return timeline

    def GetAllCellIds(self):
        """
        Returns a list of all cell ids in all frames.
        """

        cell_ids = set()
        for frame in range(len(self.frames)):
            ids = self.GetAllCellsInFrame(frame)
            if ids is None:
                continue
            cell_ids.update(ids)
        return list(cell_ids)

    def GetNumberOfFrames(self):
        return len(self.frames)
