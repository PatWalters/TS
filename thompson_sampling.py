import math
import random
import sys
from typing import List, Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm.auto import tqdm

from reagent import Reagent
from ts_logger import get_logger
from ts_utils import read_reagents


class ThompsonSampler:
    def __init__(self, known_std: float, mode="maximize", minimum_uncertainty: float = .1,
                 log_filename: Optional[str] = None):
        """
        Basic init
        :param mode: maximize or minimize
        :param minimum_uncertainty: Minimum uncertainty about the mean for the prior. We don't want to start with too
        little uncertainty about the mean if we (randomly) get initial samples which are very close together. Can set
        this higher for more exploration / diversity, lower for more exploitation.
        :param known_std: This is the "known" standard deviation for the distribution of which we are trying to estimate
        the mean. Should be proportional to the range of possible values the scoring function can produce.
        :param log_filename: Optional filename to write logging to. If None, logging will be output to stdout
        """
        # A list of lists of Reagents. Each component in the reaction will have one list of Reagents in this list
        self.reagent_lists: List[List[Reagent]] = []
        self.reaction = None
        self.evaluator = None
        self.num_prods = 0
        self.minimum_uncertainty = minimum_uncertainty
        self.known_std: float = known_std
        self.logger = get_logger(__name__, filename=log_filename)
        if mode == "maximize":
            self.pick_function = np.argmax
        elif mode == "minimize":
            self.pick_function = np.argmin
        else:
            raise ValueError(f"{mode} is not a supported argument")

    def read_reagents(self, reagent_file_list, num_to_select: Optional[int] = None):
        """
        Reads the reagents from reagent_file_list
        :param reagent_file_list: List of reagent filepaths
        :param num_to_select: Max number of reagents to select from the reagents file (for dev purposes only)
        :return: None
        """
        self.reagent_lists = read_reagents(reagent_file_list, num_to_select,
                                           minimum_uncertainty=self.minimum_uncertainty, known_std=self.known_std)
        self.num_prods = math.prod([len(x) for x in self.reagent_lists])
        self.logger.info(f"{self.num_prods:.2e} possible products")

    def get_num_prods(self) -> int:
        """
        Get the total number of possible products
        :return: num_prods
        """
        return self.num_prods

    def set_evaluator(self, evaluator):
        """
        Define the evaluator
        :param evaluator: evaluator class, must define an evaluate method that takes and RDKit molecule
        """
        self.evaluator = evaluator

    def set_reaction(self, rxn_smarts):
        """
        Define the reaction
        :param rxn_smarts: reaction SMARTS
        """
        self.reaction = AllChem.ReactionFromSmarts(rxn_smarts)

    def evaluate(self, choice_list: List[int]):
        """Evaluate a set of reagents
        :param choice_list: list of reagent ids
        :return: smiles for the reaction product, score for the reaction product
        """
        selected_reagents = []
        for idx, choice in enumerate(choice_list):
            component_reagent_list = self.reagent_lists[idx]
            selected_reagents.append(component_reagent_list[choice])
        prod = self.reaction.RunReactants([reagent.mol for reagent in selected_reagents])
        res = -1
        product_smiles = "FAIL"
        if prod:
            prod_mol = prod[0][0]  # RunReactants returns Tuple[Tuple[Mol]]
            Chem.SanitizeMol(prod_mol)
            product_smiles = Chem.MolToSmiles(prod_mol)
            res = self.evaluator.evaluate(prod_mol)
            [reagent.add_score(res) for reagent in selected_reagents]
        return product_smiles, res

    @staticmethod
    def _sample(scores: List[float]) -> float:
        """
        Creates the prior (normal distribution) from a list of scores for the reagent,
        returns a random sample from the prior.
        :param scores: list of scores previously collected for the reagent
        :return: Random sample from the prior distribution
        """
        loc = np.mean(scores)
        scale = np.std(scores)
        return np.random.normal(loc=loc, scale=scale)

    def warm_up(self, num_warmup_trials=3):
        """Warm-up phase, each reagent is sampled with num_warmup_trials random partners
        :param num_warmup_trials: number of times to sample each reagent
        """
        # get the list of reagent indices
        idx_list = list(range(0, len(self.reagent_lists)))
        # get the number of reagents for each component in the reaction
        reagent_count_list = [len(x) for x in self.reagent_lists]
        for i in idx_list:
            partner_list = [x for x in idx_list if x != i]
            # The number of reagents for this component
            current_max = reagent_count_list[i]
            # For each reagent...
            for j in tqdm(range(0, current_max), desc=f"Warmup {i + 1} of {len(idx_list)}"):
                # For each warmup trial...
                for k in range(0, num_warmup_trials):
                    current_list = [-1] * len(idx_list)
                    current_list[i] = j
                    # Randomly select reagents for each additional component of the reaction
                    for p in partner_list:
                        current_list[p] = random.randint(0, reagent_count_list[p] - 1)
                    self.evaluate(current_list)
        # initialize the mean and standard deviation for each reagent
        scores = []
        for i in range(0, len(self.reagent_lists)):
            for j in range(0, len(self.reagent_lists[i])):
                reagent = self.reagent_lists[i][j]
                reagent.init()
                scores += reagent.initial_scores
        self.logger.info(f"Top score found during warmup: {max(scores):.3f}")

    def search(self, num_cycles=25):
        """Run the search
        :param num_cycles: number of search iterations
        :return: a list of SMILES and scores
        """
        out_list = []
        for i in tqdm(range(0, num_cycles), desc="Cycle"):
            choice_list = []
            for reagent_list in self.reagent_lists:
                choice_row = []  # Create a list of scores for each reagent
                for reagent in reagent_list:
                    choice_row.append(reagent.sample())
                choice_list.append(choice_row)
            # Select a reagent for each component, according to the choice function
            pick = [self.pick_function(x) for x in choice_list]
            smiles, score = self.evaluate(pick)
            out_list.append([score, smiles])
            if i % 100 == 0:
                sorted_outlist = sorted(out_list, reverse=True)
                top_score = sorted_outlist[0][0]
                top_smiles = sorted_outlist[0][1]
                self.logger.info(f"Iteration: {i} max score: {top_score:2f} smiles: {top_smiles}")
        return out_list
