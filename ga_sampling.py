#!/usr/bin/env python

import sys
import pandas as pd
import pygad
from rdkit import Chem
import tqdm
from evaluators import DBEvaluator
import numpy as np

from baseline import read_input, unpack_input_dict
from ts_logger import get_logger


class GASampler:
    def __init__(self, json_file_name):
        self.input_dict = read_input(json_file_name)
        self.evaluator, self.rxn, self.reagent_lists = unpack_input_dict(self.input_dict)
        self.len_list = [len(x) for x in self.reagent_lists]
        self.num_reagents = len(self.len_list)
        self.gene_space = [list(range(0, x)) for x in self.len_list]
        self.solution_dict = None
        log_filename = self.input_dict.get("log_filename")
        logger = get_logger(__name__, filename=log_filename)

    def evaluate_solution(self, solution):
        reagents = [self.reagent_lists[i][int(solution[i])] for i in range(0, self.num_reagents)]
        reagent_mol_list = [x.mol for x in reagents]
        reagent_name_list = [x.reagent_name for x in reagents]
        product_name = "_".join(reagent_name_list)
        prod = self.rxn.RunReactants(reagent_mol_list)
        score = -1
        if len(prod):
            product_mol = prod[0][0]
            Chem.SanitizeMol(product_mol)
            product_smiles = Chem.MolToSmiles(product_mol)
            score = self.solution_dict.get(product_smiles)
            if score is None:
                if isinstance(self.evaluator, DBEvaluator):
                    score = self.evaluator.evaluate(product_name)
                    score = float(score)
                else:
                    score = self.evaluator.evaluate(product_mol)
                if not np.isfinite(score):
                    score = -1
                self.solution_dict[product_smiles] = [product_name, score]
            else:
                score = score[1]
        return score

    def fitness_func(self, ga_instance, solution, solution_idx):
        return self.evaluate_solution(solution)

    def get_num_evaluations(self):
        return self.evaluator.counter

    def run_ga(self):
        self.solution_dict = {}
        self.eval_count = 0
        fitness_function = self.fitness_func
        num_generations = 20
        num_parents_mating = 50
        sol_per_pop = 1000
        num_genes = len(self.gene_space)
        parent_selection_type = "sss"
        keep_parents = 1
        crossover_type = "single_point"
        mutation_type = "random"

        with tqdm.tqdm(total=num_generations) as pbar:
            ga_instance = pygad.GA(num_generations=num_generations,
                                   num_parents_mating=num_parents_mating,
                                   fitness_func=fitness_function,
                                   sol_per_pop=sol_per_pop,
                                   num_genes=num_genes,
                                   parent_selection_type=parent_selection_type,
                                   keep_parents=keep_parents,
                                   crossover_type=crossover_type,
                                   mutation_type=mutation_type,
                                   mutation_num_genes=1,
                                   gene_type=int,
                                   suppress_warnings=True,
                                   on_generation=lambda _: pbar.update(1),
                                   gene_space=self.gene_space)
            ga_instance.run()
        tmp_list = []
        for k, v in self.solution_dict.items():
            tmp_list.append([k] + v)
        solution_df = pd.DataFrame(tmp_list, columns=["SMILES", "Name", "score"])
        outfile_name = self.input_dict['results_filename']
        solution_df.to_csv(outfile_name,index=False)
        return solution_df


def main():
    #"/Users/pwalters/software/TS/examples/quinazoline_fp_sim.json"
    df_list = []
    for i in range(0,10):
        ga_sampler = GASampler(sys.argv[1])
        solution_df = ga_sampler.run_ga()
        solution_df['cycle'] = i
        df_list.append(solution_df)
        print(solution_df.sort_values("score", ascending=False))
        print(f"{ga_sampler.get_num_evaluations()} evaluations")
    pd.concat(df_list).to_csv("ten_ga_runs.csv",index=False)


if __name__ == "__main__":
    main()
