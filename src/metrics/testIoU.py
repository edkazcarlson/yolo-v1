import random
import sys
sys.path.append('..')

from config import YoloConfig
import unittest
from PIL import Image
from torch.utils import data
import torchvision
import torch
from parameterized import parameterized
from IOU import IoU

#Tests to make sure that IoU is working properly.
class TestIoU(unittest.TestCase):
    @parameterized.expand([
        [[0,0,10,10],[0,0,10,10], 1],
        [[0,0,10,10],[10,10,10,10], 0],
        [[0,0,10,10],[5,5,10,10], 25 / 175], # there are 25 units in the intersection, each box has 200 total area but 25 is overlapped so 25 / (200 - 25)
    ])
    def test_IoUSize(self, box1, box2, expectedIoU):
        outputIoU = IoU(box1, box2)
        self.assertEqual(outputIoU, expectedIoU)

    def test_randomTestsForIoU(self):
        for _ in range(100):
            box1 = self.createRandomBox()
            box2 = self.createRandomBox()
            iou1 = IoU(box1,box2)
            iou2 = IoU(box2,box1)
            self.assertEqual(iou1, iou2)

            self.assertGreaterEqual(iou1, 0)
            self.assertLessEqual(iou1, 1)


    def createRandomBox(self):
        x = random.randint(0, 100)
        y = random.randint(0, 100)
        h = random.randint(0, 100)
        w = random.randint(0, 100)
        return [x,y,h,w]


if __name__ == '__main__':
    unittest.main()