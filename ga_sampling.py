import pandas as pd
import pygad
from rdkit import Chem

from baseline import read_input, unpack_input_dict


class GASampler:
    def __init__(self, json_file_name):
        self.input_dict = read_input(json_file_name)
        self.evaluator, self.rxn, self.reagent_lists = unpack_input_dict(self.input_dict)
        self.len_list = [len(x) for x in self.reagent_lists]
        self.num_reagents = len(self.len_list)
        self.gene_space = [list(range(0, x)) for x in self.len_list]
        self.solution_dict = None

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
                score = self.evaluator.evaluate(product_mol)
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
        num_generations = 40
        num_parents_mating = 500
        sol_per_pop = 2000
        num_genes = len(self.gene_space)
        parent_selection_type = "sss"
        keep_parents = 1
        crossover_type = "single_point"
        mutation_type = "random"

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
                               gene_space=self.gene_space)
        ga_instance.run()
        tmp_list = []
        for k, v in self.solution_dict.items():
            tmp_list.append([k] + v)
        solution_df = pd.DataFrame(tmp_list, columns=["SMILES", "Name", "score"])
        return solution_df


def main():
    ga_sampler = GASampler("/Users/pwalters/software/TS/examples/quinazoline_fp_sim.json")
    solution_df = ga_sampler.run_ga()
    print(solution_df.sort_values("score", ascending=False))
    print(f"{ga_sampler.get_num_evaluations()} evaluations")


if __name__ == "__main__":
    main()
