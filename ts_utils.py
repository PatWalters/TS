import pandas as pd
from rdkit import Chem


def read_reagents(reagent_file_list, num_to_select):
    """
    Read the reagents SMILES files
    :param reagent_file_list: a list of filename
    :param num_to_select: select how many reagents to read, mostly a development function
    """
    reagent_df_list = []
    for r in reagent_file_list:
        df = pd.read_csv(r, sep=" ", names=["SMILES", "Name"])
        df = df.head(num_to_select).copy()
        # add an RDKit molecule to the dataframe
        df['mol'] = df.SMILES.apply(Chem.MolFromSmiles)
        reagent_df_list.append(df)
    return reagent_df_list
