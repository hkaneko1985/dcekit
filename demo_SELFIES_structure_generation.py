# -*- coding: utf-8 -*-
"""
@author: Hiromasa Kaneko
"""

# Generate chemical structures based on SELFIES


# settings
file_name = r'molecules_with_boiling_point'
number_of_generating_structures = 10000
max_alphabet = 0  # Maximum number of [*] to generate structures. If set 0, max_alphabet equals the number of [*] in SELFIEE of original molecules


import numpy as np
import pandas as pd
import selfies as sf
from rdkit import Chem

molecules = pd.read_csv('{0}.csv'.format(file_name), index_col=0, header=0)

smiles_lists = list(molecules.iloc[:, 0])
selfies_lists = [sf.encoder(smiles) for smiles in smiles_lists]
    
alphabet = sf.get_alphabet_from_selfies(selfies_lists)
alphabet.add('[nop]')
alphabet = list(sorted(alphabet))

pads_to_len = [sf.len_selfies(s) for s in selfies_lists]
pad_to_len = max(sf.len_selfies(s) for s in selfies_lists)
symbol_to_idx = {s: i for i, s in enumerate(alphabet)}
idx_to_symbol = {i: s for i, s in enumerate(alphabet)}
min_alphabet = pad_to_len

if max_alphabet == 0:
    max_alphabet = pad_to_len
number_of_tpyes_of_alphabet = len(symbol_to_idx)
generated_smiles_lists = []
for generation_number in range(number_of_generating_structures):
    print(generation_number + 1, '/', number_of_generating_structures)
    if min_alphabet == max_alphabet:
            number_of_alphabet = min_alphabet
    else:
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

    one_hot = sf.selfies_to_encoding(
       selfies=generated_selfies,
       vocab_stoi=symbol_to_idx,
       pad_to_len=pad_to_len,
       enc_type='one_hot'
    )
    
    one_hot_arr = np.array(one_hot)
    generated_one_hot_arr = np.array(generated_one_hot)
    if one_hot_arr.shape[0] == generated_one_hot_arr.shape[0]:
        if sum(sum(abs(one_hot_arr - generated_one_hot_arr))) == 0:
            generated_smiles = sf.decoder(generated_selfies)
            generated_mol = Chem.MolFromSmiles(generated_smiles)
            if generated_mol is not None and len(generated_smiles) != 0:
                generated_smiles_modified = Chem.MolToSmiles(generated_mol)
                generated_smiles_lists.append(generated_smiles_modified)

# delete duplications of structures
generated_smiles_lists = list(set(generated_smiles_lists))
print('Number of generated unique structures :', len(generated_smiles_lists))

generated_smiles_df = pd.DataFrame(generated_smiles_lists, columns=['SMILES'])
generated_smiles_df.to_csv('generated_smiles_from_{0}.csv'.format(file_name))
