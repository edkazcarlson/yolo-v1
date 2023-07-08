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
                percision, recall = self.calcPercisionRecall(self.model, self.data, classInd, confidenceThreshold)
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

        truePositives = []  # correctly labeled as a class
        falsePositives = [] # incorrectly assigned something to a class (model saw something where there was nothing)
        falseNegatives = [] # missed a bounding box that it should have (there was a cow in the picture, we didn't predict a box with label of cow that overlapped.)

        for _ in range(self.config.categories):
            truePositives.append(0)
            falsePositives.append(0)
            falseNegatives.append(0)

        with torch.no_grad():
            for images, labels in self.data:
                outputBatch = self.model(images)
                for label, output in zip(labels, outputBatch):# per image
                    outputBoxes = self.NonMaxSuppression(output)

                    # at this point all the boxes are above the confidence threshold and have gone through non mas suppression.
                    for classIdx in range(self.config.categoriesCount):
                        # for GT bounding box in this image of this class
                            # see if there's any prediction outputs where the IoU > threshold
                                # if there's 0, increment 1 to the FN
                                # if there's more than 1, add 1 to the TP and "assign" it to that bounding box. Make it so we can't assign this output bounding box to another GT.
                                # if there's 1+, add the best to the TP and keep the ones that didnt get assigned.
                        # at the end, any un-assigned bounding boxes are false positives and increment the counter.

        prOutput = []
        for tp, fp, fn in zip(truePositives, falsePositives, falseNegatives):
            percision = tp / (tp + fp) # of the times I guessed something was positive, how often was I correct?
            recall = tp / (tp + fn) # of the positives, how often was I able to find them.
            prOutput.append((percision, recall))

        return prOutput
                

    def NonMaxSuppression(self, output):
        """
        Returns a list of [x,y,h,w,IoU,classNum] prediction boxes for a single model prediction that has been filtered for NMS"""
        output = output.reshape(-1, self.config.cellSize)#per cell
        cellClass = output[:,self.config.categoriesStartInd:] #just the class predictions of the cell
        boundingBoxPredictions = output[:,:self.config.categoriesStartInd] #just the bounding box coord predictions 
        
        classBoundingBoxes = []
        for cellIndex in range(cellClass):
            #get the bounding boxes per class, add to classBoundingBoxes
            pass
        for classIdx in range(len(classBoundingBoxes)):
            boundingBoxes = classBoundingBoxes[classIdx]
            for boxIdx in range(len(boundingBoxes)):
                for otherBoxIdx in range(boxIdx, len(boundingBoxes)):
                    #calculate IoU, if above, mark to throw out

        return listOfBoundingBoxes


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