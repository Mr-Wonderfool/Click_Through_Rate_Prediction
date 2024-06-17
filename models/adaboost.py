import numpy as np

def indicator(judgement):
    return 1 if judgement else 0
class weakLearner:
    """### Class for weak learner

    Usage: Specify initial distribution and training sample.
        When generating hypothesis
    - call assignDistribution() to reassign distribution
    - call generateDecisionStump() to decide on feature and threshold
    - call decisionStump() to generate a decision stump function
    """
    def __init__(self, distribution: list, samples: list):
        self.TRAINNUM = len(samples)
        self.PRECISION = 10 # number of samples from uniform distribution
        self.VECLENGTH = len(samples[0][0])
        self.distribution = distribution
        self.samples = samples
        assert(len(self.distribution) == len(self.samples) and 
            abs(sum(self.distribution)-1) < 1e-6)
    def __decisionStumpClassify(self, feature, threshold):
        """
        Calculate error of the current decision stump

        Parameters
        ----------
        feature: the entry in train data
        threshold: the tested threshold

        Returns
        -------
        The error of the current decision stump
        """
        return sum(
            self.distribution[i]*indicator(
                self.samples[i][1]*(self.samples[i][0][feature]-threshold)<=0
            ) for i in range(self.TRAINNUM)
        )
    def __decisionStumpHelper_precision(self):
        """## Implementation of decision stamps

        ### Parameters
        ----------
        - distribution: a distribution on the data set
        - samples: training sample list
        ### Returns
        ------------
        - [j, s]: j-the feature selected, s-the threshold
        - reverse: True - negative decision stamp, False - positive
        decision stamp
        """
        select_j = 0
        select_s = 0
        reverse = False # whether negation would give better decision stamp
        error = [] # list if minimum error w.r.t j, contains [error_j, j, s]

        error_j = [] # each entry storing error if picking feature j
        threshold_j = [] # the threshold corresponding with j
        # loop over #features and #samples
        for j in range(self.VECLENGTH):
            rangeMin = min(self.samples, key=lambda x: x[0][j])[0][j]
            rangeMax = max(self.samples, key=lambda x: x[0][j])[0][j]
            threshold_j = [np.random.uniform(rangeMin, rangeMax) for _ in range(self.PRECISION)]
            for curr_s in threshold_j:
                error_j_s = self.__decisionStumpClassify(j, curr_s)
                error_j.append(error_j_s)
            negate_error_j = [[1-err] for err in error_j]
            min_index_pos, min_error_pos = min(enumerate(error_j), key=lambda x: x[1])
            min_index_neg, min_error_neg = min(enumerate(negate_error_j), key=lambda x: x[1])
            if min_error_pos <= min_error_neg:
                # no need to negate the decision stump
                reverse = False
                error.append([min_error_pos, j, threshold_j[min_index_pos], reverse])
            else:
                # negation gives a better decision stump
                reverse = True
                error.append([min_error_neg, j, threshold_j[min_index_neg], reverse])
            # reinitialize
            error_j = []
            threshold_j = []
        # check minimum entry in error
        min_error = min(error, key=lambda x: x[0])
        select_j = min_error[1]
        select_s = min_error[2]
        reverse = min_error[3]
        return select_j, select_s, reverse
    def __decisionStumpHelper_speed(self):
        """## Implementation of decision stamps

        ### Parameters
        ----------
        - distribution: a distribution on the data set
        - samples: training sample list
        ### Returns
        ------------
        - [j, s]: j-the feature selected, s-the threshold
        - reverse: True - negative decision stamp, False - positive
        decision stamp
        """
        select_j = 0
        select_s = 0
        reverse = False # whether negation would give better decision stamp
        error = [] # list if minimum error w.r.t j, contains [error_j, j, s]

        error_j = [] # each entry storing error if picking feature j
        threshold_j = [] # the threshold corresponding with j
        # loop over #features and #samples
        sample_copy = (self.samples).copy()
        distribution_copy = (self.distribution).copy()
        for j in range(self.VECLENGTH):
            toSort = list(zip(sample_copy, distribution_copy))
            toSort.sort(key=lambda x: x[0][0][j], reverse=True)
            sorted_samples, sorted_distribution = zip(*toSort)
            curr_s = sorted_samples[0][0][j]
            error_j_s = sum(
                sorted_distribution[i]*indicator(sorted_samples[i][1]>=0)
                for i in range(len(sorted_samples))
            )
            error_j.append(error_j_s)
            threshold_j.append(curr_s)
            for i in range(self.TRAINNUM):
                # currently taking j-th entry and choosing threshold
                curr_s = sorted_samples[i][0][j]
                # update error iteratively
                error_j_s += sorted_distribution[i]*(1-2*indicator(sorted_samples[i][1]>=0))
                threshold_j.append(curr_s)
                error_j.append(error_j_s)
            negate_error_j = [[1-err] for err in error_j]
            min_index_pos, min_error_pos = min(enumerate(error_j), key=lambda x: x[1])
            min_index_neg, min_error_neg = min(enumerate(negate_error_j), key=lambda x: x[1])
            if min_error_pos <= min_error_neg:
                # no need to negate the decision stump
                reverse = False
                error.append([min_error_pos, j, threshold_j[min_index_pos], reverse])
            else:
                # negation gives a better decision stump
                reverse = True
                error.append([min_error_neg, j, threshold_j[min_index_neg], reverse])
            # reinitialize
            error_j = []
            threshold_j = []
        # check minimum entry in error
        min_error = min(error, key=lambda x: x[0])
        select_j = min_error[1]
        select_s = min_error[2]
        reverse = min_error[3]
        return select_j, select_s, reverse
    def __sign(self, feature, threshold):
        return 1 if feature>=threshold else -1
    def generateDecisionStump(self, precision=False):
        """### Explicitly call to assign selected feature and threshold

        #### Returns
        -------------
        [j, s]: feature and threshold at current iteration
        """
        if precision:
            # loop over possible threshold using step size
            select_j, select_s, reverse = self.__decisionStumpHelper_precision()
        else:
            # use sorted version(decrease in precision)
            select_j, select_s, reverse = self.__decisionStumpHelper_speed()
        return select_j, select_s, reverse
    def assignDistribution(self, weakLearnerList: list, weightList: list):
        """Change distribution according to current class of
        weak learners

        #### Parameters
        ---------------
        - weakLearnerList: list of weak hypotheses
        - weightList: list of weights for hypotheses
        #### Returns
        ------------
        currWeight: list containing newly decided weight_i
        """
        assert(len(weakLearnerList) == len(weightList))
        currWeight = []
        for i in range(self.TRAINNUM):
            predictions = np.array([stump(self.samples, i) for stump in weakLearnerList])
            coeff = weightList @ predictions
            weight = self.samples[i][1]*coeff
            w_i = np.exp(-weight)
            currWeight.append(w_i)
        self.distribution = (currWeight/sum(currWeight)).copy()
        return currWeight
    def decisionStump(self, feature, threshold, reverse):
        def template(data, i):
            """### Generate weak hypothesis(decision stamp)

            #### Parameters
            ---------------
            i: the index for training sample
            data: train data or test data
            #### Returns
            ------------
            -sign(x_j^{(i)}-s) if reverse else sign(x_j^{(i)}-s)
            if passed in test data, then perform validation
            """
            sig = -1 if reverse else 1
            return sig*self.__sign(data[i][0][feature], threshold)
        return template
def ensemble(train_data, gamma=.2, precision=10, precise=False):
    """
    Function for adaboost(boosting with decision stump)

    Returns
    -------
    weakLearnerList: List of weakLearners(decision stump)
    weightList: List of weights
    """
    TRAINNUM = len(train_data)
    GAMMA_EXPECT = gamma
    PRECISION = precision
    PRECISE = precise
    # initial distribution
    weakLearnerList = []
    weightList = []
    init_distribution = [1/len(train_data) for _ in range(len(train_data))]
    weakHypotheses = weakLearner(init_distribution, train_data)
    ITERTIME = int(np.ceil(np.log(TRAINNUM)/(2*GAMMA_EXPECT*GAMMA_EXPECT)))
    print(f'Performing boosting for {ITERTIME} iterations\n \
        current boundary: {GAMMA_EXPECT} \n \
        current precision for threshold: {PRECISION}')
    for _ in range(ITERTIME):
        # reassign weights
        currWeight = weakHypotheses.assignDistribution(weakLearnerList, weightList)
        # contruct a weak hypothesis based on new distribution
        feature, threshold, negate = weakHypotheses.generateDecisionStump(PRECISE)
        print(f"Current feature: {feature}, current threshold: {threshold}")
        curr_decision_stump = weakHypotheses.decisionStump(feature, threshold, negate)
        # compute theta at current iteration
        correct_weight = 0
        wrong_weight = 0
        for i in range(TRAINNUM):
            if train_data[i][1]*curr_decision_stump(train_data, i) == 1:
                correct_weight += currWeight[i]
            elif train_data[i][1]*curr_decision_stump(train_data, i) == -1:
                wrong_weight += currWeight[i]
        theta_t = (1/2)*np.log(correct_weight/wrong_weight)
        weakLearnerList.append(curr_decision_stump)
        weightList.append(theta_t)
        # calculate gamma for iteration time calculation
        gamma = (1/2)-(wrong_weight/(wrong_weight+correct_weight))
        if gamma < 1e-4:
            break # cant differentiate between weak learners
        print(f'Current Boundary gamma: {gamma}')
        print(f'Current loss function: {2*np.sqrt(correct_weight*wrong_weight)}')
    return weakLearnerList, weightList

def adaboost(weakLearnerList, weightList, data):
    """
    Parameters
    ----------
    weakLearnerList, weightList returned by adaboost
    data: train data for training error prediction, 
    test data for test error prediction
    test: using test set or training set

    Returns
    -------
    Error on specified data set
    """
    dataNum = len(data)
    error = 0
    for i in range(dataNum):
        prediction = np.array([stump(data, i) for stump in weakLearnerList])
        # hypothesis being weighted combination of decision stumps
        hypothesis = np.sign(weightList @ prediction)
        real_label = data[i][1]
        error += np.abs(hypothesis - real_label) / 2
    error /= dataNum
    return error