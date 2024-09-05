from chemprop.args import PredictArgs
from chemprop.train import make_predictions


def get_mpnn_preds(rxn_smi:str) -> float:
    """Make Chemprop predictions for Ugi reaction SMILES"""

    rxn_smi = [[rxn_smi,"FC(F)(F)CO"]]

    # define args for chemprop predictor
    args = PredictArgs()
    args.features_generator =  ["rdkit_2d","ifg_drugbank_2","ugi_qmdesc_atom"]
    args.number_of_molecules = 2
    args.gpu = 0
    args.checkpoint_paths = ['../benchmark_chemprop/hyper_opt/opt_for_pred/trial_seed_60/fold_2/model_0/model.pt', '../benchmark_chemprop/hyper_opt/opt_for_pred/trial_seed_60/fold_1/model_0/model.pt', '../benchmark_chemprop/hyper_opt/opt_for_pred/trial_seed_60/fold_8/model_0/model.pt', '../benchmark_chemprop/hyper_opt/opt_for_pred/trial_seed_60/fold_6/model_0/model.pt', '../benchmark_chemprop/hyper_opt/opt_for_pred/trial_seed_60/fold_0/model_0/model.pt', '../benchmark_chemprop/hyper_opt/opt_for_pred/trial_seed_60/fold_4/model_0/model.pt', '../benchmark_chemprop/hyper_opt/opt_for_pred/trial_seed_60/fold_9/model_0/model.pt', '../benchmark_chemprop/hyper_opt/opt_for_pred/trial_seed_60/fold_7/model_0/model.pt', '../benchmark_chemprop/hyper_opt/opt_for_pred/trial_seed_60/fold_3/model_0/model.pt', '../benchmark_chemprop/hyper_opt/opt_for_pred/trial_seed_60/fold_5/model_0/model.pt']
    args.no_features_scaling = False
    args.preds_path = "./preds.csv"
    # print(rxn_smi)
    preds_result = make_predictions(args, rxn_smi)
    
    return preds_result[0][0]