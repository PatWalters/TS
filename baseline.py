from itertools import product

import pandas as pd
import useful_rdkit_utils as uru
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm.auto import tqdm
import math
from evaluators import ROCSEvaluator


def read_reagents(reagent_file_list, num_to_pick):
    reagent_df_list = []
    for r in reagent_file_list:
        df = pd.read_csv(r, sep=" ", names=["SMILES", "Name"])
        df = df.head(num_to_pick).copy()
        # add an rdkit molecule to the dataframe
        df['mol'] = df.SMILES.apply(Chem.MolFromSmiles)
        reagent_df_list.append(df)
    return reagent_df_list


def main():
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


main()
