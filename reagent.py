import numpy as np
from rdkit import Chem


class Reagent:
    __slots__ = [
        "reagent_name",
        "smiles",
        "_minimum_uncertainty",
        "initial_scores",
        "mol",
        "prior_std",
        "current_mean",
        "current_std",
        "_current_phase"
    ]

    def __init__(self, reagent_name: str, smiles: str, minimum_uncertainty: float):
        """
        Basic init
        :param reagent_name: Reagent name
        :param smiles: smiles string
        :param minimum_uncertainty: Minimum uncertainty about the mean for the prior. We don't want to start with too little
        uncertainty about the mean if we (randomly) get initial samples which are very close together. Can set this 
        higher for more exploration / diversity, lower for more exploitation. 
        """
        self.smiles = smiles
        self.reagent_name = reagent_name
        self._minimum_uncertainty = minimum_uncertainty
        self.mol = Chem.MolFromSmiles(self.smiles)
        self.initial_scores = []
        self.prior_std = 2.0
        self._current_phase = "warmup"
        self.current_mean = None
        self.current_std = None

    def add_score(self, score: float):
        """
        Either adds a score to self.initial_scores if self._current_phase == "warmup", otherwise, does the bayesian
        update of the mean and standard deviation.
        :param score: New score collected for the reagent
        :return: None
        """
        if self._current_phase == "search":
            # Then do the bayesian update
            self.current_mean = self._update_mean(score)
            self.current_std = self._update_std()
        elif self._current_phase == "warmup":
            self.initial_scores.append(score)
        else:
            raise ValueError(f"self.current_phase should be warmup or search, found {self._current_phase}")
        return

    def sample(self) -> float:
        """
        Takes a random sample from the prior distribution
        :return: sample from the prior distribution
        """
        if self._current_phase != "search":
            raise ValueError(f"Must call Reagent.init() before sampling")
        return np.random.normal(loc=self.current_mean, scale=self.current_std)

    def init(self):
        """
        After warm-up initialize self.current_mean and self.current_std
        """
        if self._current_phase != "warmup":
            raise ValueError(f"Reagent {self.reagent_name} has already been initialized.")
        elif not self.initial_scores:
            raise ValueError(f"Must collect initial scores before initializing Reagent {self.reagent_name}")

        self.current_mean = np.mean(self.initial_scores)
        self.current_std = np.std(self.initial_scores)
        # We don't want the uncertainty about the mean to be 0 (or close to zero, e.g. when all the initial samples
        # return the same score) otherwise it will be very difficult or impossible to update the mean and standard
        # deviation with new data.
        if self.current_std < self._minimum_uncertainty:
            self.current_std = self._minimum_uncertainty
        self._current_phase = "search"

    def _update_mean(self, observed_value: float) -> float:
        """
        Bayesian update to the mean
        :param observed_value: value to use to update the mean
        :return: the updated mean
        """
        numerator = self.current_std * observed_value + (self.prior_std ** 2) * self.current_mean
        denominator = self.current_std + (self.prior_std ** 2)
        return numerator / denominator

    def _update_std(self) -> float:
        """
        Bayesian update to the standard deviation
        :return: the updated standard deviation
        """
        numerator = self.current_std * (self.prior_std ** 2)
        denominator = self.current_std + (self.prior_std ** 2)
        return numerator / denominator
