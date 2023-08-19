#!/usr/bin/env python

import importlib
import json
import sys

import pandas as pd

from thompson_sampling import ThompsonSampler
from ts_logger import get_logger


def read_input(json_filename: str) -> dict:
    """
    Read input parameters from a json file
    :param json_filename: input json file
    :return: a dictionary with the input parameters
    """
    input_data = None
    with open(json_filename, 'r') as ifs:
        input_data = json.load(ifs)
        module = importlib.import_module("evaluators")
        evaluator_class_name = input_data["evaluator_class_name"]
        class_ = getattr(module, evaluator_class_name)
        evaluator_arg = input_data["evaluator_arg"]
        evaluator = class_(evaluator_arg)
        input_data['evaluator_class'] = evaluator
    return input_data


def run_ts(json_filename: str) -> None:
    """
    Perform a Thompson sampling run
    :param json_filename: Name of the json file with the input parameters
    """
    input_dict = read_input(json_filename)
    evaluator = input_dict["evaluator_class"]
    reaction_smarts = input_dict["reaction_smarts"]
    num_ts_iterations = input_dict["num_ts_iterations"]
    reagent_file_list = input_dict["reagent_file_list"]
    num_warmup_trials = input_dict["num_warmup_trials"]
    result_filename = input_dict.get("results_filename")
    ts_mode = input_dict["ts_mode"]
    known_std = input_dict.get('known_std')
    known_std = known_std if known_std is not None else 1.0
    minimum_uncertainty = input_dict.get('minimum_uncertainty')
    minimum_uncertainty = minimum_uncertainty if minimum_uncertainty is not None else 0.1
    log_filename = input_dict.get("log_filename")
    logger = get_logger(__name__, filename=log_filename)
    ts = ThompsonSampler(mode=ts_mode, known_std=known_std, minimum_uncertainty=minimum_uncertainty)
    ts.set_evaluator(evaluator)
    ts.read_reagents(reagent_file_list=reagent_file_list, num_to_select=None)
    ts.set_reaction(reaction_smarts)
    # run the warm-up phase to generate an initial set of scores for each reagent
    ts.warm_up(num_warmup_trials=num_warmup_trials)
    # run the search with TS
    out_list = ts.search(num_cycles=num_ts_iterations)
    total_evaluations = evaluator.counter
    percent_searched = total_evaluations/ts.get_num_prods() * 100
    logger.info(f"{total_evaluations} evaluations | {percent_searched:.3f}% of total")
    # write the results to disk
    out_df = pd.DataFrame(out_list, columns=["score", "SMILES"])
    if result_filename is not None:
        out_df.to_csv(result_filename, index=False)
        logger.info(f"Saved results to: {result_filename}")
    print(out_df.sort_values("score", ascending=False).drop_duplicates(subset="SMILES").head(10))


if __name__ == "__main__":
    json_file_name = sys.argv[1]
    run_ts(json_file_name)
