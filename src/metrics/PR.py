class PR:
    def __init__(self, tp, fp, fn):
        """
        Initializes a Percision Recall object with the folowing parameters
        tp: Number of true positives
        fp: number of false positives
        fn: number of false negatives
        """
        self.percision = tp / (tp + fp) # of the times I guessed something was positive, how often was I correct?
        self.recall = tp / (tp + fn) # of the positives, how often was I able to find them.