import unittest
from targetTransform import defaultTargetTransformFunc
from fakeAnnotation import getBasicAnnotation

class TestTransform(unittest.TestCase):
    def test_targetTransform(self):
        target = getBasicAnnotation()
        outputTarget = defaultTargetTransformFunc(target=target)
        self.assertEqual(outputTarget['annotation']['size']['width'], 448)
        self.assertEqual(outputTarget['annotation']['size']['height'], 448)

        firstOutputObj = outputTarget['annotation']['object'][0]
        secondOutputObj = outputTarget['annotation']['object'][1]
        self.assertEqual(firstOutputObj['bndbox']['xmin'], 50)
        self.assertEqual(firstOutputObj['bndbox']['xmax'], 100)
        self.assertEqual(firstOutputObj['bndbox']['ymin'], 33)
        self.assertEqual(firstOutputObj['bndbox']['ymax'], 100)

        self.assertEqual(secondOutputObj['bndbox']['xmin'], 100)
        self.assertEqual(secondOutputObj['bndbox']['xmax'], 150)
        self.assertEqual(secondOutputObj['bndbox']['ymin'], 70)
        self.assertEqual(secondOutputObj['bndbox']['ymax'], 110)

if __name__ == '__main__':
    test = TestTransform()
    test.test_targetTransform()