# Thompson Sampling for Virtual Screening

This repo accompanies our paper ["Thompson Sampling─An Efficient Method for Searching Ultralarge Synthesis on Demand Databases"](https://pubs.acs.org/doi/10.1021/acs.jcim.3c01790).

Thompson Sampling
is an active learning strategy that balances the tradeoff between exploitation and exploration. The code in this
repository implements Thompson Sampling as an efficient searching algorithm for screening large, un-enumerated
libraries such as [Enamine REAL SPACE](https://enamine.net/compound-collections/real-compounds/real-space-navigator).

This implementation of Thompson Sampling can be run on any un-enumerated library comprised of reactions and reagents. To
run a virtual screen using Thompson Sampling, start by selecting an un-enumerated library to search, and a screening
objective to maximize or minimize - e.g. 2D similarity, 3D similarity (such as
Openeye's [ROCS](https://docs.eyesopen.com/applications/rocs/index.html), or docking, and a query molecule (for 2D and
3D shape similarity) or target protein structure (for docking).

The algorithm begins by constructing prior distributions for the expected value of each reagent in the library. We model
the distribution of scores produced by any reagent in the library as a normal distribution, for which we are trying to
estimate the expected value (mean), assuming the standard deviation of the distribution is known. We call this the
"warmup" period, and start by randomly sampling (making and scoring a molecule with that reagent) each reagent _n_
times. The prior distribution is then constructed by taking the mean and standard deviation of the scores from the _n_
random samples.

Next we repeat the following _n_ times:

- Probabilistically select reagents by taking a random pull from the prior distribution of each reagent in the reaction.
- Select a single reagent for each component of the reaction by taking the maximum of the random pulls from all
  reagents of that component.
- Make the molecule (in-silico) and perform conformational analysis (if required for the scoring function)
- Score the molecule and calculate the bayesian update to the prior distribution

The scores and SMILES string for each molecule made and scored are saved and (optionally) written to a file.

**run_ts.py** - The main file for running Thompson Sampling via command line.

**reagent.py** - Contains the Reagent class which constructs and updates the prior distribution.

**baseline.py** - Generates brute force or random comparisons.

**evaluators.py** - Contains the evaluation functions.

**disallow_tracker.py** - Contains the class for keeping track of sampled products.

**thompson_sampling.py** - Contains the ThompsonSampling class that runs Thompson Sampling

### Setting up the environment for running Thompson Sampling

Create a new conda environment and install rdkit:
`conda create -c conda-forge -n <your-env-name> rdkit`

Activate your environment and install the rest of the requirements:
`conda activate <your-env-name>`
`pip install -r requirements.txt`

Optionally: install Openeye [toolkits](https://docs.eyesopen.com/toolkits/python/quickstart-python/install.html) to use
ROCS scoring function.

Construct a json file with the desired parameters for your run, see example json files in the `examples` directory. See
required and optional parameter explanations below.

### How to run Thompson Sampling

`python ts_main.py <path-to-json-params-file>.json`

Or try one of the example queries:

`python ts_main.py examples/amide_fp_sim.json`

or

`python ts_main.py examples/quinazoline_fp_sim.json`


### Parameters

Required params:
- `evaluator_arg`: The argument to be passed to the instantiation of the evaluator (e.g. smiles string of the query
molecule, filepath to the mol file, etc.) See the relevant `Evaluator` for more info.
- `evaluator_class_name`: Required. The scoring function to use. Must be one of: "FPEvaluator" for 2D similarity, "
MWEvaluator"for molecular weight, or "ROCSEvaluator". To use your own scoring function, implement a subclass of the
Evaluator baseclass in evaluators.py.
- `reaction_smarts`: Required. The SMARTS string for the reaction.
- `num_ts_iterations`: Required. Number of iterations of Thompson Sampling to run (usually 100 - 2000).
- `reagent_file_list`: Required. List of filepaths - one for each component of the reaction. Each file should contain the
smiles strings of valid reagents for that component.
- `num_warmup_trials`: Required. Number of times to randomly sample each reagent in the reaction. 3 is usually sufficient
for 2 component reactions, 10 is recommended for reactions with 3 or more components.
- `ts_mode`: Required. Whether to maximize or minimize the scoring function. Should be one of `maximize` or `minimize`.

Optional params:
- `results_filename`: Optional. Name of the file to output results to. If None, results will not be saved to a file.
- `known_std`: Optional. The standard deviation of the distribution for which we are trying to estimate the mean. This
should be scaled to the scoring function. For 2D similarity (for which scores can range from 0-1, we suggest using 1;
for ROCS, which ranges from 0-2, we suggest using 2, etc). If not provided, will be set to 1.0.
- `minimum_uncertainty`: Optional. Uncertainty about the expected value of the prior. If uncertainty is at or near 0 (
indicating 100% confidence in the expected value) it will be impossible to update the scoring function. Therefore, we
set a minimum uncertainty > 0 so that the prior can be updated. Increasing this number will lead to more exploration,
decreasing it will lead to more exploitation. We recommend starting with ~10% of the range of the scoring function. If
not set, a default of 0.1 is used.
- `log_filename`: Optional. Log filename to save logs to. If not set, logging will be printed to stdout.
