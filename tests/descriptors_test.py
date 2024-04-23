import unittest

import skimage.io as io
import skimage as sk
import DescriptorLib.descriptors2d as lib
from DescriptorLib import descriptor_provider as dp
import numpy as np
import skimage.measure as measure


class TestDescriptors(unittest.TestCase):
    def test_stdev(self):
        img = io.imread("tests/testdata/cell_img.tif")
        mask = io.imread("tests/testdata/cell_mask.tif")
        desc = lib.StdDev()
        res = desc.Eval(img, mask)
        self.assertTrue(np.allclose(35.51444126302579, res))

        circle_img = np.zeros((100, 100))
        rr, cc = sk.draw.disk((50, 50), 10)
        circle_img[rr, cc] = 1
        with_noise = sk.util.random_noise(circle_img)

        res = desc.Eval(circle_img, circle_img)
        self.assertTrue(np.allclose(0.0, res))

        res = desc.Eval(with_noise, circle_img)
        self.assertTrue(np.allclose(0.0, res) == False)

        res = desc.Eval(with_noise, circle_img)
        res2 = desc.Eval(with_noise, np.ones_like(circle_img))
        self.assertTrue(np.allclose(res, res2) == False)

    def test_all(self):
        img = io.imread("tests/testdata/cell_img.tif")
        mask = io.imread("tests/testdata/cell_mask.tif")
        regions = measure.regionprops(mask)

        for region in regions:
            id = region.label

            min_row, min_col, max_row, max_col = region.bbox
            region_img = np.copy(
                img[min_row:max_row + 1, min_col:max_col + 1])
            region_mask = np.copy(
                mask[min_row:max_row + 1, min_col:max_col + 1])
            region_mask = (region_mask == id).astype(int)

            res = dp.ComputeForAll(region_img, region_mask)
            self.assertTrue(len(res) != 0)


if __name__ == '__main__':
    unittest.main()
