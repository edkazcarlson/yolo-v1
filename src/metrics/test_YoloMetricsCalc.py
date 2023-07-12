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
        box1[4] = 1 #confidence

        box2[0] = .5 # corner is in center of cell
        box2[1] = .5 # corner is in center of cell
        box2[2] = 20
        box2[3] = 20
        box2[4] = .1

        singleCell = torch.cat((box1, box2, classOneHot))

        singleOutput = torch.zeros(7*7*30)
        singleOutput = singleOutput.reshape(7,7,30)
        singleOutput[0,0] = singleCell

        boundingBoxes = calc.outputToBoundingBoxes(singleOutput)

        filteredBoxes = []
        for box in boundingBoxes:
            if box[4] > 0:
                filteredBoxes.append(box) 

        self.assertEqual(len(filteredBoxes), 2)
        self.assertEqual(filteredBoxes[0][0] , (448 // 7) * .5)# pixel x
        self.assertEqual(filteredBoxes[0][1] , (448 // 7) * .5)# pixel y
        self.assertEqual(filteredBoxes[0][2] , 10)# pixel width
        self.assertEqual(filteredBoxes[0][3] , 10)# pixel height
        self.assertEqual(filteredBoxes[0][4] , 1)# confidence
        self.assertEqual(filteredBoxes[0][5] , 10)# class ind

if __name__ == '__main__':
    unittest.main()
