import numpy as np
from rdkit import Chem


class Reagent:
    __slots__ = ["reagent_name", "smiles", "scores", "mol", "prior_std", "current_mean", "current_std"]

    def __init__(self, reagent_name: str, smiles: str):
        self.smiles = smiles
        self.scores = []
        self.reagent_name = reagent_name
        self.mol = Chem.MolFromSmiles(self.smiles)
        self.prior_std = 2.0
        self.current_mean = None
        self.current_std = None

    def add_score(self, score: float):
        """
        Adds a score to self.scores
        :param score: New score to append to self.scores
        :return: None
        """
        self.scores.append(score)
        if self.current_mean and self.current_std:
            self.current_mean = self.update_mean(score)
            self.current_std = self.update_std()
        return

    def sample(self) -> float:
        """
        Takes a random sample from the prior distribution
        :return: sample from the prior distribution
        """
        return np.random.normal(loc=self.current_mean, scale=self.current_std)

    def init(self):
        """
        After warm-up initialize self.current_mean and self.current_std
        """
        self.current_mean = np.mean(self.scores)
        self.current_std = np.std(self.scores)

    def update_mean(self, observed_value: float) -> float:
        """
        Bayesian update to the mean
        :param observed_value: value to use to update the mean
        :return: the updated mean
        """
        numerator = self.current_std * observed_value + (self.prior_std ** 2) * self.current_mean
        denominator = self.current_std + (self.prior_std ** 2)
        return numerator / denominator

    def update_std(self) -> float:
        """
        Bayesian update to the standard deviation
        :return: the updated standard deviation
        """
        numerator = self.current_std * (self.prior_std ** 2)
        denominator = self.current_std + (self.prior_std ** 2)
        return numerator / denominator
