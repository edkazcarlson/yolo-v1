import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('..')
from config import YoloConfig



def retrieveMetrics(yoloModel, testData, config: YoloConfig):
    # Calculate mAP
    mAP, prs = mAP(yoloModel, testData, config)

    print(f'mAP is: {mAP}')
    # render and save percision recall curves
    for classIdx, pr in enumerate(prs):
        className = config.categories[classIdx]
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
def mAP(yoloModel, testData, config: YoloConfig, samplingPoints = [.1 * x for x in range(1,10)]):
    """
    Calculates the mean average percision 
    
    Returns:
    Mean average percision
    array of (percisionList, recallList) for each class 
    """
    apList = []

    percisionRecalls = []
    
    for classInd in range(config.categoriesCount):
        classPercisions = []
        classRecalls = []
        for confidenceThreshold in samplingPoints:
            percision, recall = calcPercisionRecall(yoloModel, testData, classInd, confidenceThreshold)
            classPercisions.append(percision)
            classRecalls.append(recall)
        
        avgClassPercision = np.mean(np.aray(classPercisions))
        apList.append(avgClassPercision)

        percisionRecalls.append((classPercisions, classRecalls))

    apList = (apList)
    mAP = np.mean(np.array(apList))

    return mAP, percisionRecalls
    


def calcPercisionRecall(yoloModel, testData, classInd: int, confidenceThreshold: float):
