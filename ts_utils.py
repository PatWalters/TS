from typing import List

import pandas as pd
from rdkit import Chem

from reagent import Reagent


def create_reagents(filename: str) -> List[Reagent]:
    # TODO: add option to only select the first n reagents
    reagent_list = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            smiles, reagent_name = line.split()
            reagent = Reagent(reagent_name=reagent_name, smiles=smiles)
            reagent_list.append(reagent)
    return reagent_list

def read_reagents(reagent_file_list, num_to_select):
    """
    Read the reagents SMILES files
    :param reagent_file_list: a list of filename
    :param num_to_select: select how many reagents to read, mostly a development function
    """
    reagent_df_list = []
    for idx, r in reagent_file_list:
        # df = pd.read_csv(r, sep=" ", names=["SMILES", "Name"])
        # df = df.head(num_to_select).copy()
        # # add an RDKit molecule to the dataframe
        # df['mol'] = df.SMILES.apply(Chem.MolFromSmiles)
        # reagent_df_list.append(df)
    return reagent_df_list
