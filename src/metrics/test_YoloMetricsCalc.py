import sys
sys.path.append('..')

from YoloMetricsCalc import YoloMetricsCalculator
from config import YoloConfig
import unittest
import torch
import torchvision

class TestMetricCalc(unittest.TestCase):
    #Tests the non max suppression of the config
    # def testNMS(self):
    #     config = YoloConfig()
    #     calc = YoloMetricsCalculator(None, None, config)
    #     fakeOutput = torch.zeros((7,7,30))
    #     firstCell = fakeOutput[0,0]
    #     firstBox = firstCell[0:5]
    #     secondBox = firstCell[5:10]
    #     firstBox[0]
    #     calc.NonMaxSuppression()

    def test_cellToBoundingBox(self):
        config = YoloConfig()
        calc = YoloMetricsCalculator(None, None, config)
        
        classOneHot = torch.zeros(config.categoriesCount)
        classOneHot[0] = .1
        classOneHot[10] = .9
        box1 = torch.zeros(5)
        box2 = torch.zeros(5)

        box1[0] = .5 # corner is in center of cell
        box1[1] = .5 # corner is in center of cell
        box1[2] = 10
        box1[3] = 10

        box1[0] = .5 # corner is in center of cell
        box1[1] = .5 # corner is in center of cell
        box1[2] = 10
        box1[3] = 10
        
        calc.outputToBoundingBoxes()


if __name__ == '__main__':
    test = TestMetricCalc()
    test.test_targetTransform()