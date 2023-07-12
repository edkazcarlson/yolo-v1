import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from IOU import IoU
from PR import PR

sys.path.append('..')
from config import YoloConfig


class YoloMetricsCalculator:
    def __init__(self, yoloModel, testData, config: YoloConfig, samplingPoints = [.1 * x for x in range(1,10)]):
        self.config = config
        self.model = yoloModel
        self.data = testData
        self.nmsIoUThreshold = .5
        self.samplingPoints = samplingPoints

    def retrieveMetrics(self):
        # Calculate mAP
        mAP, prs = mAP()

        print(f'mAP is: {mAP}')
        # render and save precision recall curves
        for classIdx in range(len(prs)):
            className = self.config.categories[classIdx]
            classPrecsisions = [x.precision for x in prs[classIdx]]
            classRecalls = [x.recalls for x in prs[classIdx]]

            classPrecsisions, classRecalls = zip(*sorted(zip(classPrecsisions, classRecalls))) # https://stackoverflow.com/questions/29876580/how-to-sort-a-list-according-to-another-list-python
            classPrecsisions.insert(0, 1) # if we have our threshold too high, our precision = 1 since everything we say is class x is class x and recall = 0 because we guess nothing. Done just to better fit graph
            classRecalls.insert(0, 0)

            plt.plot(classRecalls, classPrecsisions)
            plt.title(f'PR curve for {className}')
            plt.xlabel('Recall')
            plt.ylabel('precision')
            plt.imsave()

    # mean average precision
    def mAP(self):
        """
        Calculates the mean average precision 
        
        Returns:
        Mean average precision
        {class index -> [pr object at first confidence interval, pr object and 2nd...]}
        """
        apList = []
        
        prs = self.calcPrecisionRecall()
        
        for classIdx in range(self.config.categoriesCount):
            classPrs = prs[classIdx]
            avgClassprecision = sum([x.precision for x in classPrs])
        apList.append(avgClassprecision)


        apList = (apList)
        mAP = np.mean(np.array(apList))

        return mAP, prs

    def calcPrecisionRecall(self):
        """

        For a confidence threshold, calculate the precision and recall performance across all classes
        If the boxes IoU > confidenceThreshold, it is considered a positive prediction
        If 2 boxes with the same class prediction have an IoU > nmsIoUThreshold then the lower is removed.

        Output is in the shape
        {class index -> [pr object at first confidence interval, pr object and 2nd...]}
        """

        prOutput = {}

        with torch.no_grad():
            for images, labels in self.data:
                outputBatch = self.model(images)
                for label, modelOutput in zip(labels, outputBatch):# per image
                    nmsBoxes = self.NonMaxSuppression(modelOutput)
                    for confidenceThreshold in self.samplingPoints:
                        # filter based on the confidence threshold

                        # at this point all the boxes are above the confidence threshold and have gone through non max suppression.
                        for classIdx in range(self.config.categoriesCount):
                            TP = 0
                            FP = 0
                            FN = 0
                            # for each GT bounding box in this image of this class
                                # see if there's any pairs of prediction and GT boxes where IoU > threshold
                                    # if there's 0, increment 1 to the FN
                                    # if there's more than 1, add 1 to the TP and "assign" it to that bounding box. Make it so we can't assign this output bounding box to another GT.
                                    # if there's 1+, add the best to the TP and keep the ones that didnt get assigned.
                            # at the end, any un-assigned bounding boxes are false positives and increment the counter.

                            prOutput[classIdx].append(PR(TP, FP, FN))
        return prOutput
                

    def NonMaxSuppression(self, modelOutput):
        """
        Input: The tensor output from the model for 1 image. 
        Returns a list of [x,y,h,w,IoU,classNum] prediction boxes for a single model prediction that has been filtered for NMS
        """
        modelOutput = modelOutput.reshape(-1, self.config.cellTensorSize)#per cell
        # cellClass = modelOutput[:,self.config.categoriesStartInd:] #just the class predictions of the cell
        # boundingBoxPredictions = modelOutput[:,:self.config.categoriesStartInd] #just the bounding box coord predictions 
        
        classToBoxes = {} # class index -> list of bounding boxes

        boundingBoxes = self.outputToBoundingBoxes(modelOutput)

        for box in boundingBoxes:
            classInd = box[5]
            if classInd not in classToBoxes:
                classToBoxes[classInd] = []
            classToBoxes[classInd].append(box)

        nmsFilteredList = []


        for classInd in classToBoxes:
            boundingBoxes = classToBoxes[classInd]
            indicesToRemove = set()
            for firstBoxIdx in range(len(boundingBoxes)):
                firstBox = boundingBoxes[firstBoxIdx]
                for secondBoxIdx in range(firstBoxIdx, len(boundingBoxes)):
                    secondBox = boundingBoxes[secondBoxIdx]

                    #calculate IoU, if above, mark to throw out
                    iou = IoU(firstBox, secondBox)
                    if iou > self.nmsIoUThreshold: # there is too much overlap, decide which to mark to remove from processing
                        firstBoxConfidence = firstBox[4]
                        secondBoxConfidence = secondBox[4]
                        if firstBoxConfidence > secondBoxConfidence:
                            indicesToRemove.add(secondBoxIdx)
                        else:
                            indicesToRemove.add(firstBoxIdx)
            
            for boxIdx in range(len(boundingBoxes)):
                if boxIdx not in indicesToRemove:
                    nmsFilteredList.append(boundingBoxes[boxIdx])

        return nmsFilteredList


    def outputToBoundingBoxes(self, modelOutput: torch.tensor):
        """
        Transforms a tensor representing a prediction for a single image (7,7,30)
        into a list of [x,y,h,w,IoU,classNum]
        """
        output = []
        modelOutput = modelOutput.reshape((self.config.cellsPerAxis, self.config.cellsPerAxis, self.config.cellTensorSize))
        for cellAxisHeightInd in range(modelOutput.shape[0]):
            for cellAxisWidthInd in range(modelOutput.shape[1]):
                cell = modelOutput[cellAxisHeightInd, cellAxisWidthInd]
                cellClass = cell[self.config.categoriesStartInd:] #just the class predictions of the cell
                cellClass = torch.argmax(cellClass).item()

                boxes = cell[:self.config.categoriesStartInd]
                boxes = boxes.reshape(self.config.boxesPerCell, 5)
                for box in boxes:
                    box = box.tolist()
                    box[0] *= self.config.cellPixelLength #
                    box[0] += self.config.cellPixelLength * cellAxisHeightInd

                    box[1] *= self.config.cellPixelLength #
                    box[1] += self.config.cellPixelLength * cellAxisWidthInd

                    box.append(cellClass)
                    output.append(box)
        return output