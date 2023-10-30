import re
from glob import glob

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from rdkit import Chem


# functions to generate plots for the paper.
# It probably would have been better to generalize this into a couple of functions

def smi2cansmi(smi_in):
    mol = Chem.MolFromSmiles(smi_in)
    return Chem.MolToSmiles(mol)


def get_color(cycle):
    return 0 if cycle == "ref" else 1


def recovery_stats(df):
    ref_smiles = df.query("cycle == 'ref'").SMILES
    recovery_list = []
    for k, v in df.groupby("cycle"):
        recovery_list.append([k, len(v.query("SMILES in @ref_smiles"))])
    recovery_df = pd.DataFrame(recovery_list, columns=["cycle", "recovered"])
    return recovery_df


def plot_recovery_barplot(df, ax=None, xlabel="Cycle"):
    colors = sns.color_palette("tab10")[1:3]
    pal = [colors[0]] * 11 + [colors[1]]
    ax = sns.barplot(x="cycle", y="recovered", data=df, palette=pal, ax=ax)
    for idx, r in enumerate(df.recovered.values):
        label = str(r)
        if len(label) == 2:
            offset = 0.1
        if len(label) == 3:
            offset = 0.2
        ax.text(idx - offset, 30, label, color="white", fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of Top 100 Molecules Found")


def plot_stripplot(df, score_col, ax=None):
    df['color'] = df.cycle.apply(get_color)
    colors = sns.color_palette("tab10")[1:3]
    pal = {0: colors[1], 1: colors[0]}
    ax = sns.stripplot(x="cycle", y=score_col, data=df, hue="color", palette=pal, ax=ax)
    ax.set_xlabel("Cycle")
    ax.set_ylabel(f"{score_col} (Larger is Better)")
    ax.get_legend().set_visible(False)
    ax.set_xlabel(None)
    ax.set_xticks([])


def plot_ts(df, score_col):
    recovery_df = recovery_stats(df)
    sns.set_style('white')
    sns.set_context('talk')
    figure, axes = plt.subplots(2, 1, figsize=(20, 10), gridspec_kw={'height_ratios': [2, 1]})
    plot_stripplot(df, score_col=score_col, ax=axes[0])
    plot_recovery_barplot(recovery_df, ax=axes[1])
    plt.tight_layout()


def compile_results(file_spec, ref_file=None):
    df_list = []
    int_re = re.compile("[0-9]+")
    for filename in glob(file_spec):
        df = pd.read_csv(filename)
        cycle, warmup = [int(x) for x in int_re.findall(filename)]
        df = df.sort_values("score", ascending=False).drop_duplicates("SMILES").head(100).copy()
        df['warmup'] = warmup
        df['cycle'] = cycle
        df_list.append(df)
    combo_df = pd.concat(df_list)
    df_concat_3 = combo_df.query("warmup == 3").sort_values("score", ascending=False).drop_duplicates("SMILES").head(
        100).copy()
    df_concat_3['warmup'] = 3
    df_concat_3['cycle'] = "concat"
    df_concat_10 = combo_df.query("warmup == 10").sort_values("score", ascending=False).drop_duplicates("SMILES").head(
        100).copy()
    df_concat_10['warmup'] = 10
    df_concat_10['cycle'] = 'concat'
    df_list.append(df_concat_3)
    df_list.append(df_concat_10)
    if ref_file:
        ref_df = pd.read_csv(ref_file)
        ref_df['warmup'] = "ref"
        ref_df['cycle'] = "ref"
        df_list.append(ref_df)
    combo_df = pd.concat(df_list)
    combo_df.cycle = combo_df.cycle.astype(str)
    return combo_df


def plot_stripplot2(combo_df, include_ref=True, ax=None):
    if include_ref:
        order = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "concat", "ref"]
        hue_order = [3, 10, "ref"]
    else:
        order = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "concat"]
        hue_order = [3, 10]
    if ax is None:
        ax = sns.stripplot(x="cycle", y="score", hue="warmup", data=combo_df,
                           dodge=True, palette="tab10",
                           order=order,
                           hue_order=hue_order)
    else:
        ax = sns.stripplot(x="cycle", y="score", hue="warmup", data=combo_df,
                           dodge=True, palette="tab10",
                           order=order,
                           hue_order=hue_order, ax=ax)

    ax.legend(loc='upper left', bbox_to_anchor=(1.00, 0.75), ncol=1, title='Warmup');
    ax.set_xlabel(None)
    ax.set_xticks([])

    ax.set_ylabel("Tanimoto Coefficient (Bigger is Better)")
    handles = ax.legend_.legend_handles
    for h in handles:
        h.set_markersize(15)


def plot_recovery_barplot2(combo_df, ax=None, xlabel="Cycle"):
    ref_df = combo_df.query("warmup == 'ref'")
    match_list = []
    for i in combo_df.cycle.unique():
        if i == "ref":
            continue
        for warmup in [3, 10]:
            current_df = combo_df.query("cycle == @i and warmup == @warmup and SMILES in @ref_df.SMILES")
            match_list.append([i, len(current_df), warmup])
    match_list.append(["ref", 100, "ref"])
    match_df = pd.DataFrame(match_list, columns=['cycle', 'count', 'warmup'])
    list_ordering = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "concat", "ref"]
    match_df.cycle = pd.Categorical(match_df.cycle, list_ordering, ordered=True)
    if ax is None:
        ax = sns.barplot(x="cycle", y="count", hue="warmup", data=match_df,
                         palette="tab10")
    else:
        ax = sns.barplot(x="cycle", y="count", hue="warmup", data=match_df,
                         palette="tab10", ax=ax)
    ax.set_ylim(0, 105)
    # ax.legend(loc='upper left', bbox_to_anchor=(1.00, 0.75), ncol=1, title='Warmup')
    ax.legend_.remove()
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of Top 100 Molecules Found")
    x_val = -0.37
    labels = match_df.sort_values(['cycle', 'warmup'])["count"].values

    for i in range(0, 11):
        ax.text(x_val, 50, labels[i * 2], color="white", fontweight="bold")
        ax.text(x_val + 0.28, 50, labels[i * 2 + 1], color="white", fontweight="bold")
        x_val = x_val + 1
    ax.text(x_val + 0.48, 50, 100, color="white", fontweight="bold")
    return match_df


def plot_iteration_stripplot2(combo_df, include_ref=True, ax=None):
    if include_ref:
        order = ["2000", "5000", "10000", "50000", "100000", "ref"]
        hue_order = [3, 10, "ref"]
    else:
        order = ["2000", "5000", "10000", "50000", "100000"]
        hue_order = [3, 10]
    if ax is None:
        ax = sns.stripplot(x="iterations", y="score", hue="warmup", data=combo_df,
                           order=order,
                           dodge=True, palette="tab10",
                           hue_order=hue_order)
    else:
        ax = sns.stripplot(x="iterations", y="score", hue="warmup", data=combo_df,
                           order=order,
                           dodge=True, palette="tab10",
                           hue_order=hue_order, ax=ax)

    ax.legend(loc='upper left', bbox_to_anchor=(1.00, 0.75), ncol=1, title='Warmup');
    ax.set_xlabel(None)
    ax.set_xticks([])
    ax.set_ylabel("Tanimoto Coefficient (Bigger is Better)")
    handles = ax.legend_.legend_handles
    for h in handles:
        h.set_markersize(15)


def plot_iteration_barplot(combo_df, ax=None, xlabel="Cycle"):
    ref_df = combo_df.query("warmup == 'ref'")
    match_list = []
    for i in combo_df.cycle.unique():
        if i == "ref":
            continue
        for warmup in [3, 10]:
            current_df = combo_df.query("cycle == @i and warmup == @warmup and SMILES in @ref_df.SMILES")
            match_list.append([i, len(current_df), warmup])
    match_list.append(["ref", 100, "ref"])
    match_df = pd.DataFrame(match_list, columns=['cycle', 'count', 'warmup'])
    list_ordering = ["2000", "5000", "10000", "50000", "100000", "ref"]
    match_df.cycle = pd.Categorical(match_df.cycle, list_ordering, ordered=True)
    if ax is None:
        ax = sns.barplot(x="cycle", y="count", hue="warmup", data=match_df,
                         palette="tab10")
    else:
        ax = sns.barplot(x="cycle", y="count", hue="warmup", data=match_df,
                         palette="tab10", ax=ax)
    ax.set_ylim(0, 105)
    ax.legend(loc='upper left', bbox_to_anchor=(1.00, 0.75), ncol=1, title='Warmup')
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of Top 100 Molecules Found")
    ax.legend_.remove()

    x_val = -0.33
    labels = match_df.sort_values(['cycle', 'warmup'])["count"].values

    for i in range(0, 5):
        ax.text(x_val, 50, labels[i * 2], color="white", fontweight="bold")
        ax.text(x_val + 0.28, 50, labels[i * 2 + 1], color="white", fontweight="bold")
        x_val = x_val + 1
    ax.text(x_val + 0.52, 50, 100, color="white", fontweight="bold")

    return match_df


def compile_iteration_data(file_spec, reference_file=None):
    df_list = []
    int_re = re.compile("[0-9]+")
    for filename in glob(file_spec):
        df = pd.read_csv(filename)
        warmup, iterations = [int(x) for x in int_re.findall(filename)]
        df = df.sort_values("score", ascending=False).drop_duplicates("SMILES").head(100).copy()
        df['warmup'] = warmup
        df['iterations'] = iterations
        df_list.append(df)

    if reference_file:
        ref_df = pd.read_csv(reference_file)
        ref_df['warmup'] = "ref"
        ref_df['iterations'] = "ref"
        df_list.append(ref_df)

    combo_df = pd.concat(df_list)
    combo_df.iterations = combo_df.iterations.astype(str)
    return combo_df


def plot_random_stripplot(ref_filespec, random_filespec, ts_filespec, ax=None):
    colors = sns.color_palette("tab10")[1:4]
    int_re = re.compile("([0-9]+)")
    ref_df = pd.read_csv(ref_filespec)
    ref_df['cycle'] = 'ref'
    ref_df['method'] = 'ref'

    random_df_list = []
    for filename in glob(random_filespec):
        cycle = int_re.findall(filename)[0]
        df = pd.read_csv(filename)
        df['cycle'] = cycle
        df['method'] = 'random'
        random_df_list.append(df)
    concat_df = pd.concat(random_df_list).sort_values("score", ascending=False).drop_duplicates("SMILES").head(
        100).copy()
    concat_df['cycle'] = 'concat'
    concat_df['method'] = 'random'
    random_df_list.append(concat_df)

    ts_df_list = []
    for filename in glob(ts_filespec):
        cycle = int_re.findall(filename)[0]
        df = pd.read_csv(filename)
        df = df.sort_values("score", ascending=False).drop_duplicates("SMILES").head(100)
        df['cycle'] = cycle
        df['method'] = 'ts'
        ts_df_list.append(df)
    concat_df = pd.concat(ts_df_list).sort_values("score", ascending=False).drop_duplicates("SMILES").head(100).copy()
    concat_df['cycle'] = 'concat'
    concat_df['method'] = 'ts'
    ts_df_list.append(concat_df)

    display_df = pd.concat([ref_df] + random_df_list + ts_df_list)
    display_df.cycle = pd.Categorical(display_df.cycle,
                                      categories=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "concat", "ref"],
                                      ordered=True)
    display_df.method = pd.Categorical(display_df.method,
                                       categories=["random", "ts", "ref"], ordered=True)
    display_df = display_df.reset_index()
    ax = sns.stripplot(x="cycle", y="score", hue="method", data=display_df, ax=ax, dodge=True,
                       palette={"ref": colors[1], "ts": colors[0], "random": colors[2]})
    ax.legend(loc='upper left', bbox_to_anchor=(1.00, 0.75), ncol=1, title='Method')
    handles = ax.legend_.legend_handles
    for h in handles:
        h.set_markersize(15)
    ax.set_ylabel("Tanimoto Coefficient (Bigger is Better)")
    ax.set_xlabel(None)
    ax.set_xticks([])
    return display_df


def plot_random_recovery_barplot(combo_df, ax=None, xlabel="Replicate"):
    colors = sns.color_palette("tab10")[1:4]
    ref_df = combo_df.query("method == 'ref'")
    match_list = []
    for i in combo_df.cycle.unique():
        if i == "ref":
            continue
        for method in ["random", "ts"]:
            current_df = combo_df.query("cycle == @i and method == @method and  SMILES in @ref_df.SMILES.values")
            match_list.append([i, len(current_df), method])
    match_list.append(["ref", 100, "ref"])
    match_df = pd.DataFrame(match_list, columns=['cycle', 'count', 'method'])
    match_df.cycle = pd.Categorical(match_df.cycle,
                                    categories=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "concat", "ref"],
                                    ordered=True)
    match_df.method = pd.Categorical(match_df.method,
                                     categories=["random", "ts", "ref"], ordered=True)
    ax = sns.barplot(x="cycle", y="count", hue="method", dodge=True, data=match_df,
                     palette={"ref": colors[1], "ts": colors[0], "random": colors[2]})
    ax.legend(loc='upper left', bbox_to_anchor=(1.00, 0.75), ncol=1, title='Method')
    labels = match_df.sort_values(['cycle', 'method'], ascending=[True, True])["count"].values

    x_val = -0.39
    for i in range(0, 11):
        ax.text(x_val + 0.0, 50, labels[i * 2], color="black", fontweight="bold")
        ax.text(x_val + 0.28, 50, labels[i * 2 + 1], color="white", fontweight="bold")
        x_val = x_val + 1
    ax.text(x_val + 0.50, 50, 100, color="white", fontweight="bold")
    ax.set_ylabel("Number of Top 100 Molecules Found")
    ax.set_xlabel(xlabel)
    ax.legend_.remove()
    plt.tight_layout()
