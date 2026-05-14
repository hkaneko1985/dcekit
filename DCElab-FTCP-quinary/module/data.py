import joblib, json
import numpy as np
import pandas as pd
from functools import partial
from tqdm import tqdm
tqdm = partial(tqdm, position=0, leave=True)
from sklearn.preprocessing import OneHotEncoder
from pymatgen.core import Structure
#from mp_api.client import MPRester

'''
def data_query(mp_api_key, max_elms=3, min_elms=3, max_sites=20, include_te=False):
    The function queries data from Materials Project.

    Parameters
    ----------
    mp_api_key : str
        The API key for Mateirals Project.
    max_elms : int, optional
        Maximum number of components/elements for crystals to be queried.
        The default is 3.
    min_elms : int, optional
        Minimum number of components/elements for crystals to be queried.
        The default is 3.
    max_sites : int, optional
        Maximum number of components/elements for crystals to be queried.
        The default is 20.
    include_te : bool, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    dataframe : pandas dataframe
        Dataframe returned by MPDataRetrieval.
    
    with MPRester(mp_api_key) as mpr:
        # Query materials with specified criteria
        docs = mpr.materials.summary.search(
            energy_above_hull=(0.0, 0.08),
            fields=["material_id", "formula_pretty", "band_gap", 
                   "energy_above_hull", "structure", "spacegroup"]
        )
    
    # Convert search results to dataframe
    data_list = []
    op = tqdm(docs)
    for doc in op:
        op.set_description('converting API results to dataframe ...')
        
        structure = doc.structure
        nelements = len(structure.composition.elements)
        nsites = len(structure)
        
        # Filter by nelements and nsites criteria
        if nelements < min_elms or nelements > max_elms or nsites > max_sites:
            continue
        
        # Convert structure to CIF string
        cif_str = structure.to('cif')
        
        data_dict = {
            'material_id': doc.material_id,
            'formation_energy_per_atom': doc.energy_above_hull if hasattr(doc, 'energy_above_hull') else None,
            'band_gap': doc.band_gap if hasattr(doc, 'band_gap') else None,
            'pretty_formula': doc.formula_pretty,
            'e_above_hull': doc.energy_above_hull,
            'elements': list(structure.composition.elements),
            'cif': cif_str,
            'spacegroup_number': doc.spacegroup.number if hasattr(doc, 'spacegroup') and doc.spacegroup else None
        }
        data_list.append(data_dict)
    
    dataframe = pd.DataFrame(data_list)
    dataframe['ind'] = np.arange(len(dataframe))
    
    if include_te:
        # Read thermoelectric properties from https://datadryad.org/stash/dataset/doi:10.5061/dryad.gn001
        te = pd.read_csv('data/thermoelectric_prop.csv', index_col=0)
        te = te.dropna()
        # Get compound index that has both ground-state and thermoelectric properties
        ind = dataframe.index.intersection(te.index)
        # Concatenate thermoelectric properties to corresponding compounds
        dataframe = pd.concat([dataframe, te.loc[ind,:]], axis=1)
        dataframe['Seebeck'] = dataframe['Seebeck'].apply(np.abs)
    
    return dataframe
'''

def FTCP_represent(dataframe, max_elms=3, max_sites=20, return_Nsites=False):
    '''
    This function represents crystals in the dataframe to their FTCP representations.

    Parameters
    ----------
    dataframe : pandas dataframe
        Dataframe containing cyrstals to be converted; 
        CIFs need to be included under column 'cif'.
    max_elms : int, optional
        Maximum number of components/elements for crystals in the dataframe. 
        The default is 3.
    max_sites : int, optional
        Maximum number of sites for crystals in the dataframe.
        The default is 20.
    return_Nsites : bool, optional
        Whether to return number of sites to be used in the error calculation
        of reconstructed site coordinate matrix
    
    Returns
    -------
    FTCP : numpy ndarray
        FTCP representation as numpy array for crystals in the dataframe.

    '''
    
    # Suppress warnings
    import warnings
    warnings.filterwarnings("ignore")
    
    # Read string of elements considered in the study
    elm_str = joblib.load('data/element.pkl')
    # Build one-hot vectors for the elements
    elm_onehot = np.arange(1, len(elm_str)+1)[:,np.newaxis]
    elm_onehot = OneHotEncoder().fit_transform(elm_onehot).toarray()
    
    # Read elemental properties from atom_init.json from CGCNN (https://github.com/txie-93/cgcnn)
    with open('data/atom_init.json') as f:
        elm_prop = json.load(f)
    elm_prop = {int(key): value for key, value in elm_prop.items()}
    
    # Initialize FTCP array
    FTCP = []
    if return_Nsites:
        Nsites = []
    # Represent dataframe
    op = tqdm(dataframe.index)
    for idx in op:
        op.set_description('representing data as FTCP ...')
        
        crystal = Structure.from_str(dataframe['cif'][idx],fmt="cif")
        
        # Obtain element matrix
        elm, elm_idx = np.unique(crystal.atomic_numbers, return_index=True)
        # Sort elm to the order of sites in the CIF
        site_elm = np.array(crystal.atomic_numbers)
        elm = site_elm[np.sort(elm_idx)]
        # Zero pad element matrix to have at least 3 columns
        ELM = np.zeros((len(elm_onehot), max(max_elms, 3),))
        ELM[:, :len(elm)] = elm_onehot[elm-1,:].T
        
        # Obtain lattice matrix
        latt = crystal.lattice
        LATT = np.array((latt.abc, latt.angles))
        LATT = np.pad(LATT, ((0, 0), (0, max(max_elms, 3)-LATT.shape[1])), constant_values=0)
        
        # Obtain site coordinate matrix
        SITE_COOR = np.array([site.frac_coords for site in crystal])
        # Pad site coordinate matrix up to max_sites rows and max_elms columns
        SITE_COOR = np.pad(SITE_COOR, ((0, max_sites-SITE_COOR.shape[0]), 
                                       (0, max(max_elms, 3)-SITE_COOR.shape[1])), constant_values=0)
        
        # Obtain site occupancy matrix
        elm_inverse = np.zeros(len(crystal), dtype=int) # Get the indices of elm that can be used to reconstruct site_elm
        for count, e in enumerate(elm):
            elm_inverse[np.argwhere(site_elm == e)] = count
        SITE_OCCU = OneHotEncoder().fit_transform(elm_inverse[:,np.newaxis]).toarray()
        # Zero pad site occupancy matrix to have at least 3 columns, and max_elms rows
        SITE_OCCU = np.pad(SITE_OCCU, ((0, max_sites-SITE_OCCU.shape[0]),
                                       (0, max(max_elms, 3)-SITE_OCCU.shape[1])), constant_values=0)
        
        # Obtain elemental property matrix
        ELM_PROP = np.zeros((len(elm_prop[1]), max(max_elms, 3),))
        ELM_PROP[:, :len(elm)] = np.array([elm_prop[e] for e in elm]).T
        
        # Obtain real-space features; note the zero padding is to cater for the distance of k point in the reciprocal space
        REAL = np.concatenate((ELM, LATT, SITE_COOR, SITE_OCCU, np.zeros((1, max(max_elms, 3))), ELM_PROP), axis=0)
        
        # Obtain FTCP matrix
        recip_latt = latt.reciprocal_lattice_crystallographic
        # First use a smaller radius, if not enough k points, then proceed with a larger radius
        hkl, g_hkl, ind, _ = recip_latt.get_points_in_sphere([[0, 0, 0]], [0, 0, 0], 1.297, zip_results=False)
        if len(hkl) < 60:
            hkl, g_hkl, ind, _ = recip_latt.get_points_in_sphere([[0, 0, 0]], [0, 0, 0], 1.4, zip_results=False)
        # Drop (000)
        not_zero = g_hkl!=0
        hkl = hkl[not_zero,:]
        g_hkl = g_hkl[not_zero]
        # Convert miller indices to be type int
        hkl = hkl.astype('int16')
        # Sort hkl
        hkl_sum = np.sum(np.abs(hkl),axis=1)
        h = -hkl[:,0]
        k = -hkl[:,1]
        l = -hkl[:,2]
        hkl_idx = np.lexsort((l,k,h,hkl_sum))
        # Take the closest 59 k points (to origin)
        hkl_idx = hkl_idx[:59]
        hkl = hkl[hkl_idx,:]
        g_hkl = g_hkl[hkl_idx]
        # Vectorized computation of (k dot r) for all hkls and fractional coordinates
        k_dot_r = np.einsum('ij,kj->ik', hkl, SITE_COOR[:, :3]) # num_hkl x num_sites
        # Obtain FTCP matrix
        F_hkl = np.matmul(np.pad(ELM_PROP[:,elm_inverse], ((0, 0),
                                                           (0, max_sites-len(elm_inverse))), constant_values=0),
                          np.pi*k_dot_r.T)
        
        # Obtain reciprocal-space features
        RECIP = np.zeros((REAL.shape[0], 59,))
        # Prepend distances of k points to the FTCP matrix in the reciprocal-space features
        RECIP[-ELM_PROP.shape[0]-1, :] = g_hkl
        RECIP[-ELM_PROP.shape[0]:, :] = F_hkl
        
        # Obtain FTCP representation, and add to FTCP array
        FTCP.append(np.concatenate([REAL, RECIP], axis=1))
        
        if return_Nsites:
            Nsites.append(len(crystal))
    FTCP = np.stack(FTCP)
    
    if not return_Nsites:
        return FTCP
    else:
        return FTCP, np.array(Nsites)