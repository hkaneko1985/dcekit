import joblib, os
import numpy as np
from tqdm import tqdm
from ase.io import write
from ase import spacegroup
from pymatgen.ext.matproj import MPRester

def get_info(ftcp_designs, 
             max_elms=3, 
             max_sites=20, 
             elm_str=joblib.load('data/element.pkl'),
             to_CIF=True,
             check_uniqueness=True,
             mp_api_key=None,
             ):
    
    '''
    This function gets chemical information for designed FTCP representations, 
    i.e., formulas, lattice parameters, site fractional coordinates.
    (decoded sampled latent points/vectors).

    Parameters
    ----------
    ftcp_designs : numpy ndarray
        Designed FTCP representations for decoded sampled latent points/vectors.
        The dimensions of the ndarray are number of designs x latent dimension.
    max_elms : int, optional
        Maximum number of components/elements for designed crystals. 
        The default is 3.
    max_sites : int, optional
        Maximum number of sites for designed crystals.
        The default is 20.
    elm_str : list of element strings, optional
        A list of element strings containing elements considered in the design.
        The default is from "elements.pkl".
    to_CIF : bool, optional
        Whether to output CIFs to "designed_CIFs" folder. The default is true.
    check_uniqueness : bool, optional
        Whether to check the uniqueness of the designed composition is contained in the Materials Project.
    mp_api_key : str, optional
        The API key for Mateirals Project. Required if check_uniqueness is True. 
        The default is None.
    

    Returns
    -------
    pred_formula : list of predicted sites
        List of predicted formulas as lists of predicted sites.
    pred_abc : numpy ndarray
        Predicted lattice constants, abc, of designed crystals; 
        Dimensions are number of designs x 3
    pred_ang : numpy ndarray
        Predicted lattice angles, alpha, beta, and gamma, of designed crystals; 
        Dimensions are number of designs x 3
    pred_latt : numpy ndarray
        Predicted lattice parameters (concatenation of pred_abc and pred_ang);
        Dimensions are number of designs x 6
    pred_site_coor : list
        List of predicted site coordinates, of length number of designs;
        The component site coordinates are in numpy ndarray of number_of_sites x 3
    ind_unique : list
        Index for unique designs. Will only be returned if check_uniqueness is True.
    
    '''
    
    Ntotal_elms = len(elm_str)
    # Get predicted elements of designed crystals
    pred_elm = np.argmax(ftcp_designs[:, :Ntotal_elms, :max_elms], axis=1)
    
    def get_formula(ftcp_designs, ):
        
        # Initialize predicted formulas
        pred_for_array = np.zeros((ftcp_designs.shape[0], max_sites))
        pred_formula = []
        # Get predicted site occupancy of designed crystals
        pred_site_occu = ftcp_designs[:, Ntotal_elms+2+max_sites:Ntotal_elms+2+2*max_sites, :max_elms]
        # Zero non-max values per site in the site occupancy matrix
        temp = np.repeat(np.expand_dims(np.max(pred_site_occu, axis=2), axis=2), max_elms, axis=2)
        pred_site_occu[pred_site_occu < temp]=0
        # Put a threshold to zero empty sites (namely, the sites due to zero padding)
        pred_site_occu[pred_site_occu < 0.05] = 0
        # Ceil the max per site to ones to obtain one-hot vectors
        pred_site_occu = np.ceil(pred_site_occu)
        # Get predicted formulas
        for i in range(len(ftcp_designs)):
            pred_for_array[i] = pred_site_occu[i].dot(pred_elm[i])
            
            if np.all(pred_for_array[i] == 0):
                pred_formula.append([elm_str[0]])
            else:
                temp = pred_for_array[i]
                temp = temp[:np.where(temp>0)[0][-1]+1]
                temp = temp.tolist()
                pred_formula.append([elm_str[int(j)] for j in temp])
        return pred_formula
    
    pred_formula = get_formula(ftcp_designs)
    # Get predicted lattice of designed crystals
    pred_abc = ftcp_designs[:, Ntotal_elms, :3]
    pred_ang = ftcp_designs[:, Ntotal_elms+1,:3]
    pred_latt = np.concatenate((pred_abc, pred_ang), axis=1)
    # Get predicted site coordinates of designed crystals
    pred_site_coor = []
    pred_site_coor_ = ftcp_designs[:, Ntotal_elms+2:Ntotal_elms+2+max_sites, :3]
    for i, c in enumerate(pred_formula):
        Nsites = len(c)
        pred_site_coor.append(pred_site_coor_[i, :Nsites, :])
    
    if check_uniqueness:
        assert mp_api_key != None, "You need a mp_api_key to check the uniqueness of designed CIFs!"
        # Obtain unique designed compositions
        mpr = MPRester(mp_api_key)
        ind = []
        op = tqdm(range(len(pred_formula)))
        for i in op:
            op.set_description("Checking uniqueness of designed compostions in the Materials Project database")
            query = mpr.get_data(''.join(pred_formula[i])) 
            if not query:
                ind.append(i)
    else:
        # A dummy index
        ind = list(np.arange(len(pred_formula)))
    
    if to_CIF:
        os.makedirs('designed_CIFs', exist_ok=True)
        
        op = tqdm(ind)
        for i, j in enumerate(op):
            op.set_description("Writing designed crystals as CIFs")
            
            try:
                crystal = spacegroup.crystal(pred_formula[j],
                                             basis=pred_site_coor[j],
                                             cellpar=pred_latt[j])
                write('designed_CIFs/'+str(i)+'.cif', crystal)
            except:
                pass
    
    if check_uniqueness:
        ind_unique = ind
        return pred_formula, pred_abc, pred_ang, pred_latt, pred_site_coor, ind_unique
    else:
        return pred_formula, pred_abc, pred_ang, pred_latt, pred_site_coor