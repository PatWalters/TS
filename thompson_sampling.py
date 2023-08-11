import random
from typing import List, Optional

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm.auto import tqdm

from evaluators import ROCSEvaluator
from evaluators import FPEvaluator
from reagent import Reagent
from ts_utils import read_reagents

import math


# This is just a placeholder implementation
# TODO: Refactor so that the code does not need to re-compute the prior each time

class ThompsonSampler:
    def __init__(self, mode="maximize"):
        # A list of lists of Reagents. Each component in the reaction will have one list of Reagents in this list
        self.reagent_lists: List[List[Reagent]] = []
        self.reaction = None
        self.evaluator = None
        if mode == "maximize":
            self.pick_function = np.argmax
        elif mode == "minimize":
            self.pick_function = np.argmin
        else:
            raise ValueError(f"{mode} is not a supported argument")

    def read_reagents(self, reagent_file_list, num_to_select: Optional[int] = None):
        self.reagent_lists = read_reagents(reagent_file_list, num_to_select)
        num_prods = math.prod([len(x) for x in self.reagent_lists])
        print(f"{num_prods:.2e} possible products")

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
                    # Ranomdly select reagents for each additional component of the reaction
                    for p in partner_list:
                        current_list[p] = random.randint(0, reagent_count_list[p] - 1)
                    self.evaluate(current_list)

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
            out_list.append([smiles, score])
        return out_list


def main():
    num_iterations = 500
    reagent_file_list = ["data/aminobenzoic_ok.smi", "data/primary_amines_ok.smi", "data/carboxylic_acids_ok.smi"]
    ts = ThompsonSampler()
    fp_evaluator = FPEvaluator("COC(=O)[C@@H](CC(=O)O)n1c(C[C@H](O)C(=O)OC)nc2c(OC)cccc2c1=O")
    ts.set_evaluator(fp_evaluator)
    # rocs_evaluator = ROCSEvaluator("data/2chw_lig.sdf")
    # ts.set_evaluator(rocs_evaluator)
    ts.read_reagents(reagent_file_list=reagent_file_list, num_to_select=None)
    quinazoline_rxn_smarts = "N[c:4][c:3]C(O)=O.[#6:1][NH2].[#6:2]C(=O)[OH]>>[C:2]c1n[c:4][c:3]c(=O)n1[C:1]"
    ts.set_reaction(quinazoline_rxn_smarts)
    ts.warm_up(num_warmup_trials=10)
    out_list = ts.search(num_cycles=num_iterations)
    out_df = pd.DataFrame(out_list, columns=["SMILES", "score"])
    out_df.to_csv("ts_results.csv", index=False)
    print(out_df.sort_values("score", ascending=False).drop_duplicates(subset="SMILES").head(10))


if __name__ == "__main__":
    main()
