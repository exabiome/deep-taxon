from abc import ABCMeta, abstractmethod

import torch

class AbstractMetric(object, metaclass=ABCMeta):

    @abstractmethod
    def score(self, target, output):
        pass


    def __call__(self, target, output):
        return self.score(target, output)


class NCorrect(AbstractMetric):

    def score(self, target, output):
        _, pred = torch.max(output.data, 1)
        n_correct = (pred == target).sum().item()
        return n_correct


class NeighborNCorrect(AbstractMetric):

    def __init__(self, difile):
        self.knn = difile.get_knn_classifier()

    def score(self, target, output):
        dat = output.data
        labels = target.numpy()
        pred = self.knn.predict(dat)
        n_correct = (pred == labels).sum()
        return n_correct

class Accuracy(AbstractMetric):

    def __init__(self, ncorrect):
        self.ncorrect = ncorrect

    def score(self, target, output):
        n_correct = self.ncorrect(target, output)
        return n_correct/output.shape[0]

