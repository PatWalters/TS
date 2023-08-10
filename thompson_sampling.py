import random
from typing import List

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm.auto import tqdm

from evaluators import ROCSEvaluator
from evaluators import FPEvaluator
from ts_utils import read_reagents

import math


# This is just a placeholder implementation
# TODO: Refactor so that the code does not need to re-compute the prior each time

class ThompsonSampler:
    def __init__(self, mode="maximize"):
        self.reagent_df_list = []
        self.reaction = None
        self.evaluator = None
        # Weight list contains a list of scores for each reagent and each component. For example, a reaction with two
        # components with 2 reagents each might look like the following, assuming each reagent has been sampled 3x:
        # [ [ [1., 2., .5], [1.2, 1.7, .3] ], [ [1.1, 0.4, .02], [.8, .9, 1.] ] ]
        self.weight_list = []
        if mode == "maximize":
            self.pick_function = np.argmax
        elif mode == "minimize":
            self.pick_function = np.argmin
        else:
            raise ValueError(f"{mode} is not a supported argument")

    def read_reagents(self, reagent_file_list, num_to_select=-1):
        self.reagent_df_list = read_reagents(reagent_file_list, num_to_select)
        num_prods = math.prod([len(x) for x in self.reagent_df_list])
        print(f"{num_prods:.2e} possible products")
        # initialize empty weigh lists
        for df in self.reagent_df_list:
            self.weight_list.append([[] for _ in range(len(df))])

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

    def evaluate(self, choice_list):
        """Evaluate a set of reagents
        :param choice_list: list of reagent ids
        :return: smiles for the reaction product, score for the reaction product
        """
        reagent_mol_list = []
        for i in range(0, len(self.reagent_df_list)):
            reagent_df = self.reagent_df_list[i]
            choice = choice_list[i]
            reagent_mol_list.append(reagent_df.mol.values[choice])
        prod = self.reaction.RunReactants(reagent_mol_list)
        res = -1
        product_smiles = "FAIL"
        if len(prod):
            prod_mol = prod[0][0]
            Chem.SanitizeMol(prod_mol)
            product_smiles = Chem.MolToSmiles(prod_mol)
            res = self.evaluator.evaluate(prod_mol)
            for i, c in enumerate(choice_list):
                self.weight_list[i][c].append(res)
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
        idx_list = list(range(0, len(self.reagent_df_list)))
        # get the number of reagents
        reagent_count_list = [len(x) for x in self.reagent_df_list]
        for i in idx_list:
            partner_list = [x for x in idx_list if x != i]
            current_max = reagent_count_list[i]
            for j in tqdm(range(0, current_max), desc=f"Warmup {i + 1} of {len(idx_list)}"):
                for k in range(0, num_warmup_trials):
                    current_list = [-1] * len(idx_list)
                    current_list[i] = j
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
            for r_list in self.weight_list:
                choice_row = []
                for wt in r_list:
                    choice_row.append(self._sample(scores=wt))
                choice_list.append(choice_row)
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
    #rocs_evaluator = ROCSEvaluator("data/2chw_lig.sdf")
    #ts.set_evaluator(rocs_evaluator)
    ts.read_reagents(reagent_file_list,-1)
    quinazoline_rxn_smarts = "N[c:4][c:3]C(O)=O.[#6:1][NH2].[#6:2]C(=O)[OH]>>[C:2]c1n[c:4][c:3]c(=O)n1[C:1]"
    ts.set_reaction(quinazoline_rxn_smarts)
    ts.warm_up(num_warmup_trials=10)
    out_list = ts.search(num_cycles=num_iterations)
    out_df = pd.DataFrame(out_list, columns=["SMILES", "score"])
    out_df.to_csv("ts_results.csv", index=False)
    print(out_df.sort_values("score", ascending=False).drop_duplicates(subset="SMILES").head(10))


if __name__ == "__main__":
    main()
