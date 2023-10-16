import unittest

import skimage.io as io
import skimage as sk
import DescriptorLib.descriptors as lib
import numpy as np


class TestStatisticalDescriptors(unittest.TestCase):
    def test_stdev(self):
        img = io.imread("tests/testdata/cell_img.tif")
        mask = io.imread("tests/testdata/cell_mask.tif")
        res = lib.StdDev(img, mask)
        self.assertTrue(np.allclose(35.51444126302579, res))

        circle_img = np.zeros((100,100))
        rr,cc = sk.draw.disk((50, 50), 10)
        circle_img[rr,cc] = 1
        with_noise = sk.util.random_noise(circle_img)
        
        res = lib.StdDev(circle_img, circle_img)
        self.assertTrue(np.allclose(0.0, res))

        res = lib.StdDev(with_noise, circle_img)
        self.assertTrue(np.allclose(0.0, res) == False)

        res = lib.StdDev(with_noise, circle_img)
        res2 = lib.StdDev(with_noise, np.ones_like(circle_img))
        self.assertTrue(np.allclose(res, res2) == False)
    


if __name__ == '__main__':
    unittest.main()
