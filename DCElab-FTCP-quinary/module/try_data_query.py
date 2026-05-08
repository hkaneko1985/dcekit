# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 10:10:08 2025

@author: eri_m
"""

import numpy as np
import pandas as pd
from mp_api.client import MPRester
from tqdm import tqdm
from functools import partial
import pickle
from pymatgen.core import Structure

tqdm = partial(tqdm, position=0, leave=True)

mp_api_key = 'mFnrpK7kHHooGFO3KTwP347xQ5nts0Kd'
max_elms=4
min_elms=3
max_sites=40
include_te=False

with MPRester(mp_api_key) as mpr:
    docs = mpr.materials.summary.search(
        num_elements=(min_elms, max_elms),
        num_sites=(1, max_sites),
        energy_above_hull=(None, 0.08),
        fields=[
            "material_id",
            "formation_energy_per_atom",
            "band_gap",
            "formula_pretty",
            "energy_above_hull",
            "elements",
            "structure",
            "symmetry"
        ],
        chunk_size=1000  # to avoid overload
    )

    # Convert list of docs to DataFrame
    df = pd.DataFrame([{
        "material_id": doc.material_id,
        "formation_energy_per_atom": doc.formation_energy_per_atom,
        "band_gap": doc.band_gap,
        "pretty_formula": doc.formula_pretty,
        "e_above_hull": doc.energy_above_hull,
        "elements": doc.elements,
        "cif": doc.structure,  # pymatgen Structure object
        "spacegroup.number": doc.symmetry.number if doc.symmetry else None
    } for doc in docs])

    df['ind'] = np.arange(len(df))

# 熱電特性 Seebeck 係数を MP から取得したデータと結合する
if include_te:
    te = pd.read_csv('data/thermoelectric_prop.csv', index_col=0).dropna()
    df.index = list(df['material_id'])
    ind = df.index.intersection(te.index)
    df = pd.concat([df, te.loc[ind]], axis=1)

df['cif'] = df['cif'].apply(lambda s: s.to(fmt="cif"))
df.to_csv('data_query_3_4_elements_property.csv')

