import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision


sys.path.append('..')
from config import YoloConfig


class YoloMetricsCalculator:
    def __init__(self, yoloModel, testData, config: YoloConfig):
        self.config = config
        self.model = yoloModel
        self.data = testData
        self.nmsIoUThreshold = .5

    def retrieveMetrics(self):
        # Calculate mAP
        mAP, prs = mAP()

        print(f'mAP is: {mAP}')
        # render and save percision recall curves
        for classIdx, pr in enumerate(prs):
            className = self.config.categories[classIdx]
            percision = pr[0]
            recall = pr[1]
            percision, recall = zip(*sorted(zip(percision, recall))) # https://stackoverflow.com/questions/29876580/how-to-sort-a-list-according-to-another-list-python
            percision.insert(0, 1) # if we have our threshold too high, our percision = 1 since everything we say is class x is class x and recall = 0 because we guess nothing. Done just to better fit graph
            recall.insert(0, 0)

            plt.plot(recall, percision)
            plt.title(f'PR curve for {className}')
            plt.xlabel('Recall')
            plt.ylabel('Percision')
            plt.imsave()

    # mean average percision
    def mAP(self, samplingPoints = [.1 * x for x in range(1,10)]):
        """
        Calculates the mean average percision 
        
        Returns:
        Mean average percision
        array of (percisionList, recallList) for each class 
        """
        apList = []

        percisionRecalls = []
        
        for classInd in range(self.config.categoriesCount):
            classPercisions = []
            classRecalls = []
            for confidenceThreshold in samplingPoints:
                percision, recall = calcPercisionRecall(self.model, self.data, classInd, confidenceThreshold)
                classPercisions.append(percision)
                classRecalls.append(recall)
            
            avgClassPercision = np.mean(np.aray(classPercisions))
            apList.append(avgClassPercision)

            percisionRecalls.append((classPercisions, classRecalls))

        apList = (apList)
        mAP = np.mean(np.array(apList))

        return mAP, percisionRecalls

    def calcPercisionRecall(self, confidenceThreshold: float):
        """

        For a confidence threshold, calculate the percision and recall performance across all classes
        If the boxes IoU > confidenceThreshold, it is considered a positive prediction
        If 2 boxes with the same class prediction have an IoU > nmsIoUThreshold then the lower is removed.

        Output is in the shape
        [(percision, recall for class 0), (percision, recall for class 1), etc]
        """

        truePositives = [] 
        falsePositives = []
        trueNegatives = []
        for _ in range(self.config.categories):
            truePositives.append(0)
            falsePositives.append(0)
            trueNegatives.append(0)

        with torch.no_grad():
            for images, labels in self.data:
                output = self.model(images)
                output = output.reshape(-1, self.config.cellSize)#per cell
                cellClass = output[:,self.config.categoriesStartInd:] #just the class predictions of the cell
                boundingBoxPredictions = output[:,:self.config.categoriesStartInd] #just the bounding box coord predictions 
                
                for cellIndex in range(cellClass):
                    self.cellToBoundingBox
                #calculate the predictions > threshold

    def cellToBoundingBox(self, cellClass: int, cellBoxes: torch.tensor, confidenceThreshold: float):
        """
        Transforms a self.config.categoriesStartInd sized tensor representing the x bounding boxes and the class prediction 
        into a list of [x,y,h,w,IoU,classNum]"""
        output = []
        cellBoxes = cellBoxes.reshape(self.config.boxesPerCell, 5)
        for box in cellBoxes:
            if cellBoxes[self.config.indexOfIoU] > confidenceThreshold:
                box = box.tolist()
                box.append(cellClass)
                output.append(cellClass)
        return output