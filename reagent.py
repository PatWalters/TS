import numpy as np
from rdkit import Chem


class Reagent:
    __slots__ = ["reagent_name", "smiles", "scores", "mol"]

    def __init__(self, reagent_name: str, smiles: str):
        self.smiles = smiles
        self.scores = []
        self.reagent_name = reagent_name
        self.mol = Chem.MolFromSmiles(self.smiles)

    @property
    def mean(self) -> float:
        """
        Returns the mean of self.scores
        """
        if not self.scores:
            raise ValueError(f"Must add scores to self.scores before accessing the mean")
        return np.mean(self.scores)

    @property
    def std(self) -> float:
        """
        Returns the standard deviation of self.scores
        """
        if not self.scores:
            raise ValueError(f"Must add scores to self.scores before accessing the standard deviation")
        return np.std(self.scores)

    def add_score(self, score: float):
        """
        Adds a score to self.scores
        :param score: New score to append to self.scores
        :return: None
        """
        # TODO: once we have the bayesian update to a prior distribution we can just directly re-compute the mean and
        #  standard deviation rather than appending to the list
        self.scores.append(score)
        return

    def sample(self) -> float:
        """
        Takes a random sample from the prior distribution
        :return: sample from the prior distribution
        """
        return np.random.normal(loc=self.mean, scale=self.std)
