# -*- coding: utf-8 -*-
"""
@author: Hiromasa Kaneko
"""

import numpy as np
import pandas as pd
import selfies as sf
from rdkit import Chem

number_of_generating_structures = 10000  # Number of generating structures (target)
max_alphabet = 50  # Maximum number of [*] to generate structures. If set 0, max_alphabet equals the number of [*] in SELFIEE of original molecules
molecules = pd.read_csv(r'molecules_with_boiling_point.csv', index_col=0, header=0)  # Original molecules
#molecules = pd.read_csv(r'molecules_with_logS.csv', index_col=0, header=0)

min_alphabet = 9
smiles_lists = list(molecules.iloc[:, 0])
# SMILES to SELFIES
selfies_lists = [sf.encoder(smiles) for smiles in smiles_lists]
# Preparation for structure generation
alphabet = sf.get_alphabet_from_selfies(selfies_lists)
alphabet.add('[nop]')
alphabet = list(sorted(alphabet))
if max_alphabet == 0:
    max_alphabet = max(sf.len_selfies(s) for s in selfies_lists)
symbol_to_idx = {s: i for i, s in enumerate(alphabet)}
idx_to_symbol = {i: s for i, s in enumerate(alphabet)}
number_of_tpyes_of_alphabet = len(symbol_to_idx)
# Structure generation
generated_smiles_lists = []
for generation_number in range(number_of_generating_structures):
    print(generation_number + 1, '/', number_of_generating_structures)
    number_of_alphabet = np.random.randint(min_alphabet, max_alphabet)
    
    generated_one_hot = []
    for i in range(number_of_alphabet - 1):
        one_hot_tmp = np.zeros(number_of_tpyes_of_alphabet, dtype='int64')
        one_hot_tmp[np.random.randint(0, number_of_tpyes_of_alphabet - 1)] = 1
        generated_one_hot.append(list(one_hot_tmp))
    one_hot_tmp = np.zeros(number_of_tpyes_of_alphabet, dtype='int64')
    one_hot_tmp[-1] = 1
    
    generated_one_hot.append(list(one_hot_tmp))
    generated_selfies = sf.encoding_to_selfies(
       encoding=generated_one_hot,
       vocab_itos=idx_to_symbol,
       enc_type='one_hot'
    )
    generated_smiles = sf.decoder(generated_selfies)
    generated_mol = Chem.MolFromSmiles(generated_smiles)
    if generated_mol is not None and len(generated_smiles) != 0:
        generated_smiles_modified = Chem.MolToSmiles(generated_mol)
        generated_smiles_lists.append(generated_smiles_modified)
# Delete duplications of structures
generated_smiles_lists = list(set(generated_smiles_lists))
print('Number of generated structures :', len(generated_smiles_lists))
# Save results
generated_smiles_df = pd.DataFrame(generated_smiles_lists, columns=['SMILES'])
generated_smiles_df.to_csv('generated_smiles.csv')
