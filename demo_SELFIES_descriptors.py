# -*- coding: utf-8 -*-
"""
@author: Hiromasa Kaneko
"""

# Calculate SELFIES descriptors

# settings
file_name = r'molecules_with_boiling_point'


import numpy as np
import pandas as pd
import selfies as sf

dataset = pd.read_csv('{0}.csv'.format(file_name), index_col=0, header=0)

smileses = dataset.iloc[:, 0]
if dataset.shape[1] > 1:
    properties = dataset.iloc[:, 1:]
selfies_lists = []

for smiles in smileses:
    selfies = sf.encoder(smiles)
    selfies_lists.append(selfies)
    
alphabet = sf.get_alphabet_from_selfies(selfies_lists)
alphabet.add('[nop]')
alphabet = list(sorted(alphabet))

pad_to_len = max(sf.len_selfies(s) for s in selfies_lists)
symbol_to_idx = {s: i for i, s in enumerate(alphabet)}
idx_to_symbol = {i: s for i, s in enumerate(alphabet)}

selfies_descriptors = np.zeros([dataset.shape[0], pad_to_len * len(symbol_to_idx)])
for index, selfies in enumerate(selfies_lists):
    one_hot = sf.selfies_to_encoding(
       selfies=selfies,
       vocab_stoi=symbol_to_idx,
       pad_to_len=pad_to_len,
       enc_type='one_hot'
    )    
    selfies_descriptors[index, :] = np.array(one_hot).ravel()

selfies_descriptors = pd.DataFrame(selfies_descriptors, index=dataset.index)
if dataset.shape[1] > 1:
    selfies_descriptors = pd.concat([properties, selfies_descriptors], axis=1)
print(selfies_descriptors)

# save
selfies_descriptors.to_csv(r'SELFIES_descriptors_of_{0}.csv'.format(file_name))
