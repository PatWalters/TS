import numpy as np
from rdkit import Chem


class Reagent:
    __slots__ = [
        "reagent_name",
        "smiles",
        "min_uncertainty",
        "initial_scores",
        "mol",
        "known_var",
        "current_mean",
        "current_std",
        "current_phase"
    ]

    def __init__(self, reagent_name: str, smiles: str, minimum_uncertainty: float, known_std: float):
        """
        Basic init
        :param reagent_name: Reagent name
        :param smiles: smiles string
        :param minimum_uncertainty: Minimum uncertainty about the mean for the prior. We don't want to start with too
        little uncertainty about the mean if we (randomly) get initial samples which are very close together. Can set
        this higher for more exploration / diversity, lower for more exploitation.
        :param known_std: This is the "known" standard deviation for the distribution of which we are trying to
        estimate the mean. Should be proportional to the range of possible values the scoring function can produce.
        """
        self.smiles = smiles
        self.reagent_name = reagent_name
        self.min_uncertainty = minimum_uncertainty
        self.mol = Chem.MolFromSmiles(self.smiles)
        self.initial_scores = []
        self.known_var = known_std ** 2
        self.current_phase = "warmup"
        self.current_mean = None
        self.current_std = None

    def add_score(self, score: float):
        """
        Either adds a score to self.initial_scores if self._current_phase == "warmup", otherwise, does the bayesian
        update of the mean and standard deviation.
        :param score: New score collected for the reagent
        :return: None
        """
        if self.current_phase == "search":
            current_var = self.current_std ** 2
            # Then do the bayesian update
            self.current_mean = self._update_mean(current_var=current_var, observed_value=score)
            self.current_std = self._update_std(current_var=current_var)
        elif self.current_phase == "warmup":
            self.initial_scores.append(score)
        else:
            raise ValueError(f"self.current_phase should be warmup or search, found {self.current_phase}")
        return

    def sample(self) -> float:
        """
        Takes a random sample from the prior distribution
        :return: sample from the prior distribution
        """
        if self.current_phase != "search":
            raise ValueError(f"Must call Reagent.init() before sampling")
        return np.random.normal(loc=self.current_mean, scale=self.current_std)

    def init(self):
        """
        After warm-up initialize self.current_mean and self.current_std
        """
        if self.current_phase != "warmup":
            raise ValueError(f"Reagent {self.reagent_name} has already been initialized.")
        elif not self.initial_scores:
            raise ValueError(f"Must collect initial scores before initializing Reagent {self.reagent_name}")

        self.current_mean = np.mean(self.initial_scores)
        self.current_std = np.std(self.initial_scores)
        # We don't want the uncertainty about the mean to be 0 (or close to zero, e.g. when all the initial samples
        # return the same score) otherwise it will be very difficult or impossible to update the mean and standard
        # deviation with new data.
        if self.current_std < self.min_uncertainty:
            self.current_std = self.min_uncertainty
        self.current_phase = "search"

    def _update_mean(self, current_var: float, observed_value: float) -> float:
        """
        Bayesian update to the mean
        :param current_var: The current variance
        :param observed_value: value to use to update the mean
        :return: the updated mean
        """
        numerator = current_var * observed_value + self.known_var * self.current_mean
        denominator = current_var + self.known_var
        return numerator / denominator

    def _update_std(self, current_var: float) -> float:
        """
        Bayesian update to the standard deviation
        :param current_var: The current variance
        :return: the updated standard deviation
        """
        numerator = current_var * self.known_var
        denominator = current_var + self.known_var
        return np.sqrt(numerator / denominator)
