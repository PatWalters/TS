#!/usr/bin/env python

import heapq
import math
from itertools import product

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm.auto import tqdm

from ts_main import read_input, parse_input_dict
from ts_utils import read_reagents


def keep_largest(items, n):
    """Keeps the n largest items in a list, designed to work with a list of [score,SMILES]
    :param items: the list of items to keep
    :param n: the number of items to keep
    :return: list of the n largest items
    """
    heap = []
    for item in items:
        if len(heap) < n:
            heapq.heappush(heap, item)
        else:
            if item[0] > heap[0][0]:
                heapq.heapreplace(heap, item)
    return heap


def unpack_input_dict(input_dict, num_to_select=None):
    """ Unpack the input dictionary and create the Evaluator object
    :param input_dict:
    :param num_to_select:
    :return:
    """
    if input_dict.get("evaluator_class") is None:
        parse_input_dict(input_dict)
    evaluator = input_dict["evaluator_class"]
    reaction_smarts = input_dict["reaction_smarts"]
    reagent_file_list = input_dict["reagent_file_list"]
    rxn = AllChem.ReactionFromSmarts(reaction_smarts)
    reagent_lists = read_reagents(reagent_file_list, num_to_select)
    return evaluator, rxn, reagent_lists


def setup_baseline(json_filename, num_to_select=None):
    """ Common code for baseline methods, reads JSON input and creates necessary objects
    :param json_filename: JSON file with configuration options
    :param num_to_select: number of reagents to use with exhaustive search. Set to a lower values for development.
    Setting to None uses all reagents.
    :return: evaluator class, RDKit reaction, list of lists with reagents
    """
    input_dict = read_input(json_filename, num_to_select)
    return unpack_input_dict(input_dict)


def random_baseline(input_dict, num_trials, outfile_name, num_to_save=100, ascending_output=False):
    """ Randomly combine reagents
    :param json_filename: JSON file with parameters
    :param num_trials: number of molecules to generate
    :param num_to_save: number of molecules to save to the output csv file
    :param ascending_output: save the output in ascending order
    """
    score_list = []
    evaluator, rxn, reagent_lists = unpack_input_dict(input_dict)
    num_reagents = len(reagent_lists)
    len_list = [len(x) for x in reagent_lists]
    total_prods = math.prod(len_list)
    print(f"{total_prods:.2e} products")
    for i in tqdm(range(0, num_trials)):
        reagent_mol_list = []
        for j in range(0, num_reagents):
            reagent_idx = np.random.randint(0, len_list[j] - 1)
            reagent_mol_list.append(reagent_lists[j][reagent_idx].mol)
        prod = rxn.RunReactants(reagent_mol_list)
        if len(prod):
            product_mol = prod[0][0]
            Chem.SanitizeMol(product_mol)
            score = evaluator.evaluate(product_mol)
            product_smiles = Chem.MolToSmiles(product_mol)
            score_list = keep_largest(score_list + [[score, product_smiles]], num_to_save)
    score_df = pd.DataFrame(score_list, columns=["score", "SMILES"])
    score_df.sort_values(by="score", ascending=ascending_output).to_csv(outfile_name, index=False)


def exhaustive_baseline(json_filename, num_to_select=None, num_to_save=100, invert_score=False):
    """ Exhaustively combine all reagents
    :param json_filename: JSON file with parameters
    :param num_to_select: Number of reagents to use, set to a lower number for development.  Set to None to use all.
    :param num_to_save: number of molecules to save to the output csv file
    :param ascending_output: save the output in ascending order
    """
    score_list = []
    evaluator, rxn, reagent_lists = setup_baseline(json_filename, num_to_select)
    len_list = [len(x) for x in reagent_lists]
    total_prods = math.prod(len_list)
    print(f"{total_prods:.2e} products")
    for reagents in tqdm(product(*reagent_lists), total=total_prods):
        reagent_mol_list = [x.mol for x in reagents]
        prod = rxn.RunReactants(reagent_mol_list)
        if len(prod):
            product_mol = prod[0][0]
            Chem.SanitizeMol(product_mol)
            product_smiles = Chem.MolToSmiles(product_mol)
            score = evaluator.evaluate(product_mol)
            if invert_score:
                score = score * -1.0
            score_list = keep_largest(score_list + [[score, product_smiles]], num_to_save)
    score_df = pd.DataFrame(score_list, columns=["score", "SMILES"])
    score_df.sort_values(by="score", ascending=False).to_csv("exhaustive_scores.csv", index=False)
