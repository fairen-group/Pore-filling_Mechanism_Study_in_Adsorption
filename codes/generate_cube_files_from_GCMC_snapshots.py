import numpy as np
from ase import Atoms
from ase.io import write


def make_prob_density(pdb_filename, atm, grid_size):
    """
    Calculates a 3d probability density map of the adsorbate molecules within
    the cubic unit cell using snapshots taken from GCMC calculations.

    Parameters:
    - pdb_filename: str, path to the pdb file, taken from GCMC calculations 
                    (e.g. RASPA), containing snapshots of the production cycles.
    - atm: str, atomic symbol of the adsorbate
    - grid_size: int, the size of the probability density np.ndarray

    Returns:
    - prob_density: np.ndarray, a 3d probability distribution of adsorbate
                    molecules within the cubic unit cell
    """
    with open(pdb_filename) as f:
        for line in f:
            if line.startswith("CRYST1"):
                a = float(line[6:15])
                b = float(line[15:24])
                c = float(line[24:33])
                break
    print(a,b,c)

    coords = []
    with open(pdb_filename) as f:
        in_model = False
        for line in f:
            if line.startswith("MODEL"):
                in_model = True
            elif line.startswith("ENDMDL"):
                in_model = False
            elif in_model and line.startswith("ATOM"):
                element = line[76:78].strip()
                if element == atm:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])
    coords = np.array(coords)
    
    edges = [
        np.linspace(0, a, grid_size + 1),
        np.linspace(0, b, grid_size + 1),
        np.linspace(0, c, grid_size + 1)
    ]
    hist, edges = np.histogramdd(coords, bins=edges)
    
    prob_density = hist / hist.sum()
    
    return prob_density

def local_average_3d(array, n):
    """
    For each element in a 3D numpy array, compute the average of all elements
    whose indices are within n units in each dimension.

    Parameters:
    - array: np.ndarray of shape (X, Y, Z)
    - n: int, the max distance in each index direction for sliding window

    Returns:
    - avg_array: np.ndarray of same shape as input array with averaged values
    """
    from numpy.lib.stride_tricks import sliding_window_view
    
    padded = np.pad(array, pad_width=n, mode='constant', constant_values=0)

    window_size = 2 * n + 1
    windows = sliding_window_view(padded, (window_size, window_size, window_size))
    
    avg_array = windows.mean(axis=(-3, -2, -1))

    mask = np.ones_like(array)
    padded_mask = np.pad(mask, pad_width=n, mode='constant', constant_values=0)
    mask_windows = sliding_window_view(padded_mask, (window_size, window_size, window_size))
    counts = mask_windows.sum(axis=(-3, -2, -1))

    avg_array = (windows.sum(axis=(-3, -2, -1))) / counts

    return avg_array

def get_filtered_prob_density(pdb_filename_before, pdb_filename_after, grid_size, cell_a, atm, padding_n):
    """
    Calculates the probability density map of the adsorbate molecules inside
    the cubic unit cell, before and after the step as well as the raw and
    filtered form of the difference probability density, converts them into
    cube files, and saves the cube files in the current directory.
    """
    prob_density_before = make_prob_density(pdb_filename_before, atm, grid_size)
    prob_density_after = make_prob_density(pdb_filename_after, atm, grid_size)

    prob_density_before = (prob_density_before - prob_density_before.min()) / (prob_density_before.max() - prob_density_before.min())
    prob_density_after = (prob_density_after - prob_density_after.min()) / (prob_density_after.max() - prob_density_after.min())

    prob_density_diff = prob_density_after - prob_density_before
    prob_density_diff_filtered = local_average_3d(prob_density_diff, padding_n)

    # print(prob_density_diff.min())
    # print(prob_density_diff.max())

    cell = np.eye(3) * cell_a
    atoms = Atoms(cell=cell, pbc=True)
    
    write(f'prob_density_{atm}_before_step.cube', atoms, data=prob_density_before)
    write(f'prob_density_{atm}_after_step.cube', atoms, data=prob_density_after)
    write(f'prob_density_{atm}_difference.cube', atoms, data=prob_density_diff)
    write(f'prob_density_{atm}_difference_filtered.cube', atoms, data=prob_density_diff_filtered)

    return 0

#******************************************************************************
#*************************** USER INPUT PARAMETERS ****************************

grid_size = 200         # Number of bins per axis
cell_a = 50.256         # Unit cell "a" length in Angstrom
atm = 'Ar'              # Atomic symbol of the adsorbate
padding_n = 15          # distance in each index direction for sliding window

pdb_filename_before = "Movie_CU-6_Cr_P1_pacman_1.1.1_87.000000_6000.000000_allcomponents.pdb"  # Snapshots before the step
pdb_filename_after = "Movie_CU-6_Cr_P1_pacman_1.1.1_87.000000_10000.000000_allcomponents.pdb"  # Snapshots after the step

#******************************************************************************

prob_density_diff_filtered = get_filtered_prob_density(pdb_filename_before, pdb_filename_after, grid_size, cell_a, atm, padding_n)
print('Done!')



