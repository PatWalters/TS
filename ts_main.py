#!/usr/bin/env python

import importlib
import json
import sys

import pandas as pd

from thompson_sampling import ThompsonSampler


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
    ts_mode = input_dict["ts_mode"]

    ts = ThompsonSampler(mode=ts_mode, known_std=1.0, minimum_uncertainty=.1)
    ts.set_evaluator(evaluator)
    ts.read_reagents(reagent_file_list=reagent_file_list, num_to_select=None)
    ts.set_reaction(reaction_smarts)
    # run the warm-up phase to generate an initial set of scores for each reagent
    ts.warm_up(num_warmup_trials=num_warmup_trials)
    # run the search with TS
    out_list = ts.search(num_cycles=num_ts_iterations)
    total_evaluations = evaluator.get_num_evaluations()
    percent_searched = total_evaluations/ts.get_num_prods() * 100
    print(f"{total_evaluations} evaluations | {percent_searched:.3f}% of total")
    # write the results to disk
    out_df = pd.DataFrame(out_list, columns=["score", "SMILES"])
    out_df.to_csv("ts_results.csv", index=False)
    print(out_df.sort_values("score", ascending=False).drop_duplicates(subset="SMILES").head(10))


if __name__ == "__main__":
    json_file_name = sys.argv[1]
    run_ts(json_file_name)
