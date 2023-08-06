import math
from itertools import product

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm.auto import tqdm

from evaluators import ROCSEvaluator
from ts_utils import read_reagents


def exhaustive_baseline():
    rocs_eval = ROCSEvaluator("data/2chw_lig.sdf")

    reagent_file_list = ["data/aminobenzoic_100.smi", "data/primary_amines_100.smi", "data/carboxylic_acids_100.smi"]
    quinazoline_smarts = "N[c:4][c:3]C(O)=O.[#6:1][NH2].[#6:2]C(=O)[OH]>>[C:2]c1n[c:4][c:3]c(=O)n1[C:1]"
    rxn = AllChem.ReactionFromSmarts(quinazoline_smarts)
    reagent_df_list = read_reagents(reagent_file_list, 100)
    mol_lol = [x.mol for x in reagent_df_list]
    score_list = []
    len_list = [len(x) for x in reagent_df_list]
    total_prods = math.prod(len_list)
    for reagents in tqdm(product(*mol_lol), total=total_prods):
        prod = rxn.RunReactants(reagents)
        if len(prod):
            product_mol = prod[0][0]
            Chem.SanitizeMol(product_mol)
            score = rocs_eval.evaluate(product_mol)
            score_list.append(score)
    print(max(score_list))


def random_baseline():
    num_trials = 1000
    num_reagents_to_read = 100
    rocs_eval = ROCSEvaluator("data/2chw_lig.sdf")

    reagent_file_list = ["data/aminobenzoic_100.smi", "data/primary_amines_100.smi", "data/carboxylic_acids_100.smi"]
    quinazoline_smarts = "N[c:4][c:3]C(O)=O.[#6:1][NH2].[#6:2]C(=O)[OH]>>[C:2]c1n[c:4][c:3]c(=O)n1[C:1]"
    rxn = AllChem.ReactionFromSmarts(quinazoline_smarts)
    reagent_df_list = read_reagents(reagent_file_list, num_reagents_to_read)

    mol_lol = [x.mol for x in reagent_df_list]
    len_list = [len(x) for x in reagent_df_list]
    num_reagents = len(len_list)
    product_score_list = []
    for i in tqdm(range(0, num_trials)):
        reagent_mol_list = []
        for j in range(0, num_reagents):
            reagent_idx = np.random.randint(0, len_list[j] - 1)
            reagent_mol_list.append(mol_lol[j][reagent_idx])
        prod = rxn.RunReactants(reagent_mol_list)
        if len(prod):
            product_mol = prod[0][0]
            Chem.SanitizeMol(product_mol)
            product_smiles = Chem.MolToSmiles(product_mol)
            score = rocs_eval.evaluate(product_mol)
            product_score_list.append([product_smiles, score])
    out_df = pd.DataFrame(product_score_list, columns=["SMILES", "score"])
    out_df.to_csv("random_results.csv", index=False)
    print(out_df.sort_values("score", ascending=False).drop_duplicates(subset="SMILES").head(10))


if __name__ == "__main__":
    random_baseline()
