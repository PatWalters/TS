#!/usr/bin/env python

import json
import os
import sys

TS_BASE_DIR = ".."
sys.path.append(TS_BASE_DIR)

from ts_main import run_ts, parse_input_dict
from baseline import random_baseline
from tqdm.auto import tqdm


# Benchmark functions used to generate data for the figures for the paper

def compare_warmup_cycles(input_dict, base_name, num_cycles=10, warmup_trial_list=[3, 10]):
    for i in tqdm(range(0, num_cycles)):
        for num_warmup_trials in warmup_trial_list:
            input_dict["num_warmup_trials"] = num_warmup_trials
            input_dict[
                "results_filename"] = f"benchmark_data/ts_replicate_{base_name}_{i + 1}_warmup_{num_warmup_trials}.csv"
            run_ts(input_dict, hide_progress=True)


def run_iteration_trials(input_dict, output_prefix):
    warmup_trial_list = [3, 10]
    ts_iteration_list = [2000, 5000, 10000, 50000, 100000]
    for num_ts_iterations in tqdm(ts_iteration_list):
        for num_warmup_trials in warmup_trial_list:
            input_dict["num_warmup_trials"] = num_warmup_trials
            input_dict["num_ts_iterations"] = num_ts_iterations
            input_dict[
                "results_filename"] = f"benchmark_data/ts_warmup_{output_prefix}_{num_warmup_trials}_iterations_{num_ts_iterations}.csv"
            run_ts(input_dict, hide_progress=True)


def run_random_trials(input_dict, num_random_cycles=10, num_warmup_trials=10, num_ts_iterations=50000):
    input_dict["num_warmup_trials"] = num_warmup_trials
    input_dict["num_ts_iterations"] = num_ts_iterations
    for i in range(0, num_random_cycles):
        random_baseline(input_dict, num_trials=num_ts_iterations, outfile_name=f"benchmark_data/ts_random_{i + 1}.csv")


quinazoline_json = """{
"reagent_file_list": [
        "TS_BASE_DIR/data/aminobenzoic_ok.smi",
        "TS_BASE_DIR/data/primary_amines_500.smi",
        "TS_BASE_DIR/data/carboxylic_acids_500.smi"
    ],
    "reaction_smarts": "N[c:4][c:3]C(O)=O.[#6:1][NH2].[#6:2]C(=O)[OH]>>[C:2]c1n[c:4][c:3]c(=O)n1[C:1]",
    "num_warmup_trials": 10,
    "num_ts_iterations": 50000,
    "evaluator_class_name": "FPEvaluator",
    "evaluator_arg": {"query_smiles" : "CCc1cccc2c(=O)n(C3CNC3)c([C@@H](C)N)nc12"},
    "ts_mode": "maximize",
    "log_filename": "ts_logs.txt",
    "results_filename": "ts_results.csv"
}""".replace("TS_BASE_DIR", TS_BASE_DIR)


def main():
    os.makedirs("benchmark_data", exist_ok=True)
    quinazoline_dict = json.loads(quinazoline_json)
    parse_input_dict(quinazoline_dict)
    run_iteration_trials(quinazoline_dict, "quinazoline")
    compare_warmup_cycles(quinazoline_dict, "quinazoline")
    run_random_trials(quinazoline_dict)


if __name__ == "__main__":
    main()
