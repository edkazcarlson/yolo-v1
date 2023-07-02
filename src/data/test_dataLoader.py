import sys
sys.path.append('..')

from config import YoloConfig
import unittest
from dataLoaders import VOCDataset
from targetTransform import defaultTargetTransformFunc
from PIL import Image
from fakeAnnotation import getBasicAnnotation
from torch.utils import data
import torchvision
import torch

class fakeDataset(data.Dataset):
	def __init__(self, iterable):
		self.iterable = iterable

	def __len__(self):
		return len(self.iterable)

	def __getitem__(self, index):
		return self.iterable[index]

class TestVocDataset(unittest.TestCase):
	def test_loadFile(self):
		iterable = []
		transformedTarget =  defaultTargetTransformFunc(getBasicAnnotation())
		imageT = torchvision.transforms.ToTensor()
		transformedImage = imageT(Image.new('RGB', size = (448, 448)))
		fakeUnderlyingDataPoint = (transformedImage, transformedTarget)
		iterable.append(fakeUnderlyingDataPoint)
		fakeUnderlyingDataset = fakeDataset(iterable)
		# for x in fakeUnderlyingDataset:
		# 	print(x)

		dataSet = data.DataLoader(VOCDataset(fakeUnderlyingDataset, YoloConfig()))
		self.assertEqual(len(dataSet), 1)
		for x in dataSet:
			image, target = x
			self.assertEqual(image.shape[2], 448) #height
			self.assertEqual(image.shape[3], 448) #width
			
			self.assertEqual(target.shape[0], 1)
			self.assertEqual(target.shape[1], 7)
			self.assertEqual(target.shape[2], 7)
			self.assertEqual(target.shape[3], 30)

			self.assertNotEqual(target.sum(), 0) # there is something in the target set properly.

if __name__ == '__main__':
    test = TestVocDataset()
    test.test_loadFile()