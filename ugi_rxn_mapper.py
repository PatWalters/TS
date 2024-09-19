"""Use this script like `from ugi_mapper_new import ugi_rxn_mapper` """

from rdkit import Chem
from rdkit.Chem import AllChem

# from rdkit.RDLogger import logger
# logger = logger()
# logger.setLevel(4)

from typing import List, Dict, Any, Union, Tuple, Optional
from collections import defaultdict
import re


"""reaction using amine"""
retro_ugi_4c4c_smarts_1 = "[C;!R:6](=[O:7])[N;!R:3]([#6:4])[CH1,H2;!R:5][C+0;!R:1](=[OH0+0:8])[NH1+0;!R:2]>>[C-:1]#[NH0+:2].[NX3H2:3][#6:4].[CX3H1;!$(*O);!$(*S);!$(*N):5](=[O]).[CX3:6](=[O:7])[OX2H1:8]"
retro_ugi_4c4c_reaction_1 = AllChem.ReactionFromSmarts(retro_ugi_4c4c_smarts_1)

"""reaction using ammonium hydroxide"""
retro_ugi_4c4c_smarts_2 = "[C;!R:6](=[O:7])[NH1;!R:3][CH1,H2;!R:5][C+0;!R:1](=[OH0+0:8])[NH1+0;!R:2]>>[C-:1]#[NH0+:2].[NX3H3:3].[CX3H1;!$(*O);!$(*S);!$(*N):5](=[O]).[CX3:6](=[O:7])[OX2H1:8]"
retro_ugi_4c4c_reaction_2 = AllChem.ReactionFromSmarts(retro_ugi_4c4c_smarts_2)


reactants_smarts_list_1 = [
    "[C-:1]#[N+:2]",
    # '[NX3H2:3][#6;!$(C=[C,O,N,S]);!$(C#*):4]',
    "[NH2,NH3+:3][CX4,c]",
    "[C:5](=[O:6])",
    "[CX3:7](=[O:8])[OX2H1:9]",
]

reactants_smarts_list_2 = [
    "[C-:1]#[N+:2]",
    "[NH3+0,NH4+1:3]",
    "[C:5](=[O:6])",
    "[CX3:7](=[O:8])[OX2H1:9]",
]

product_smarts_1 = "[C:7](=[O:8])[N:3]([#6:4])[C:5][C+0:1](=[OH0+0:9])[NH:2]"
product_smarts_2 = "[C:7](=[O:8])[N:3][C:5][C+0:1](=[OH0+0:9])[NH:2]"

reactants_list_1 = [
    Chem.MolFromSmarts(reactant) for reactant in reactants_smarts_list_1
]
reactants_list_2 = [
    Chem.MolFromSmarts(reactant) for reactant in reactants_smarts_list_2
]

product_1 = Chem.MolFromSmarts(product_smarts_1)
product_2 = Chem.MolFromSmarts(product_smarts_2)


def ugi_rxn_mapper(smiles_list: list) -> list:
    """
    This is main function of this file. Import this function to get atom mapped ugi rxn smiles from product smiles

    Parameters
    ----------
    smiles_list : list
        list of product smiles

    Returns
    -------
    ugi_rxn_smi: list
        list of atom mapped ugi rxn smiles

    """

    ugi_rxn_smi = []

    for smi in smiles_list:
        atom_mapped_ugi_rxn_smi = _ugi_mapping(smi)
        ugi_rxn_smi.append(atom_mapped_ugi_rxn_smi)

    return ugi_rxn_smi


"""The following functions are used in ugi_rxn_mapper"""


def _mapProdAtom(prod: str) -> str:

    prod = Chem.MolFromSmiles(prod)
    for idx, atom in enumerate(prod.GetAtoms()):
        atom.SetAtomMapNum(idx + 1)
    return Chem.MolToSmiles(prod)


def _getMapDict(
    patt_mol: Chem.Mol,
    targ_mol: Chem.Mol,
    reverse: bool = False,
    a: int = 1000,
    b: int = 10,
) -> defaultdict:

    hit_at = targ_mol.GetSubstructMatches(patt_mol)
    if not hit_at:
        return None

    hit_smarts = Chem.MolFragmentToSmarts(targ_mol, atomsToUse=hit_at[0])
    hit_mol = Chem.MolFromSmarts(hit_smarts)

    patt_smiles = Chem.MolToSmiles(patt_mol)
    patt_tags = re.findall(r"\:(\d+)", patt_smiles)
    re_pattern = re.compile(r"[A-GI-Z\(\)=#+-]")
    smiles_pattern = "".join(re_pattern.findall(patt_smiles))

    for _ in range(a):
        hit_mol_smiles_set = {
            Chem.MolToSmiles(hit_mol, doRandom=True) for _ in range(b)
        }
        hit_mol_smiles_set.update(Chem.MolToSmiles(hit_mol))
        matched_smiles = [
            smiles
            for smiles in hit_mol_smiles_set
            if "".join(re_pattern.findall(smiles)) == smiles_pattern
        ]
        if matched_smiles:
            matched_smiles = matched_smiles[0]
            targ_tags = re.findall(r"\:(\d+)", matched_smiles)
            mapping_dict = defaultdict(
                int,
                {
                    int(target): int(template)
                    for template, target in zip(patt_tags, targ_tags)
                },
            )
            if reverse:
                mapping_dict = defaultdict(
                    int,
                    {
                        int(target): int(template)
                        for template, target in zip(targ_tags, patt_tags)
                    },
                )
            return mapping_dict

    return None


def _getRunReactants(
    prod: Chem.Mol, rxn: AllChem.ChemicalReaction
) -> Optional[Tuple[Chem.Mol]]:

    reactants = rxn.RunReactants((prod,))
    if len(reactants) == 0:
        return None
    reactants = reactants[0]
    return reactants


def _processProduct(prod_mapped: str) -> Optional[str]:

    prod = Chem.MolFromSmiles(prod_mapped)
    reactants_1 = _getRunReactants(prod, retro_ugi_4c4c_reaction_1)
    reactants_2 = _getRunReactants(prod, retro_ugi_4c4c_reaction_2)
    if reactants_1 is None and reactants_2 is None:
        print(
            f"Input product {Chem.MolToSmiles(prod)} cannot match the given reaction SMARTS"
        )
        return None

    if reactants_1 is not None:
        rxn_smarts = _processReaction(prod, reactants_1, product_1, reactants_list_1)
        if rxn_smarts:
            return rxn_smarts

    if reactants_2 is not None:
        rxn_smarts = _processReaction(prod, reactants_2, product_2, reactants_list_2)
        if rxn_smarts:
            return rxn_smarts

    print(
        f"Input product {Chem.MolToSmiles(prod)} cannot match the given reaction SMARTS"
    )
    return None


def _mapReactAtom(
    reacts: List[Chem.Mol], reacts_sub: List[Chem.Mol], map_dict: Dict[int, int]
) -> List[Chem.Mol]:

    for react, react_sub in zip(reacts, reacts_sub):

        if react.HasSubstructMatch(react_sub):
            hit_at = react.GetSubstructMatches(react_sub)
            if not hit_at:
                continue

            nomap_count = 300
            for idx in hit_at[0]:
                atom = react.GetAtomWithIdx(idx)
                if atom.GetAtomMapNum() == 0:
                    atom.SetAtomMapNum(nomap_count)
                    nomap_count += 1
            sub_map_dict = _getMapDict(react_sub, react)
            if sub_map_dict is None:
                continue

            for idx in hit_at[0]:
                atom = react.GetAtomWithIdx(idx)
                atom_map = atom.GetAtomMapNum()
                # print(map_dict[sub_map_dict[atom_map]])
                atom.SetAtomMapNum(map_dict[sub_map_dict[atom_map]])

    return reacts


def _rxnMols2Smiles(react_mols: List[Chem.Mol], prod_mol: Chem.Mol) -> Tuple[str, str]:

    reaction = AllChem.ChemicalReaction()
    for react in react_mols:
        reaction.AddReactantTemplate(react)
    reaction.AddProductTemplate(prod_mol)
    return AllChem.ReactionToSmarts(reaction)


def _processReaction(prod, reactants, product, reactants_list) -> Optional[str]:

    map_dict = _getMapDict(product, prod, reverse=True)
    if map_dict is None:
        return None

    mapped_reactants = _mapReactAtom(reactants, reactants_list, map_dict)
    return _rxnMols2Smiles(mapped_reactants, prod)


def _ugi_mapping(smi):

    mapped_prod_smi = _mapProdAtom(smi)
    rxn_smarts = _processProduct(mapped_prod_smi)
    return rxn_smarts
