import os

from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors
import numpy as np
from Mold2_pywrapper.mold2_wrapper import Mold2

def cal_descriptors(mols):
    des_name = [i[0] for i in Descriptors._descList]
    des_cal = MoleculeDescriptors.MolecularDescriptorCalculator(des_name)
    cal_results = [des_cal.CalcDescriptors(m) for m in mols]
    des_pd = pd.DataFrame((np.array(cal_results)), columns=des_name)
    return des_pd


def find_bits(m1, m2, name):
    diff_bits = []
    for em, (b1, b2) in enumerate(zip(m1, m2)):
        # if np.around(b1, 2) != np.around(b2, 2):
        if np.around(b1, 2) != np.around(b2, 2):
            # print(np.around(b1, 2), " ",np.around(b2, 2))
            diff_bits.append(em)
    print(name," ", diff_bits)
    print(len(diff_bits))
    return diff_bits


def main():
    file_read = pd.read_excel("3A4same.xlsx")
    smiles = [file_read.Smiles.values.tolist(), file_read.Training_smiles.values.tolist()]
    m = Mold2()
    question_molecular = []
    for i, j in zip(smiles[0], smiles[1]):
        mol1 = Chem.MolFromSmiles(i)
        mol2 = Chem.MolFromSmiles(j)
        print(Chem.MolToInchiKey(mol1)==Chem.MolToInchiKey(mol2))
        print(Chem.MolToInchiKey(mol1), Chem.MolToInchiKey(mol2))
        continue
        # MACCS
        maccs1 = AllChem.GetMACCSKeysFingerprint(mol1)
        maccs2 = AllChem.GetMACCSKeysFingerprint(mol2)


        # ECFP4
        ecfp41 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2)
        ecfp42 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2)
        # assert ecfp41 != ecfp42, "Data leakage!"


        # FCFP4
        fcfp41 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, useFeatures=True)
        fcfp42 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, useFeatures=True)
        assert fcfp41 != fcfp42, "Data leakage!"
        assert maccs1 != maccs2, "Data leakage!"
        find_bits(fcfp41, fcfp42, "fcfp4")

        # RDKit descriptors
        rd_des = cal_descriptors([mol1, mol2])
        assert all(np.array(rd_des)[0] == np.array(rd_des)[1]) is False, "Data leakage!"
        same_d = find_bits(np.array(rd_des)[0], np.array(rd_des)[1], "fcfp41")
        cal_name = []
        for k in same_d:
            cal_name += [rd_des.columns[k]]
        print(cal_name)
        Mold2 descriptors
        mol_des = m.calculate([mol1, mol2])
        if not (all(np.array(mol_des)[0] == np.array(mol_des)[1]) is False):
            question_molecular.append(Chem.MolToSmiles(mol1))
        find_bits(np.array(mol_des, dtype=float)[0], np.array(mol_des,dtype=float)[1], "fcfp4")
    # print(question_molecular)

main()