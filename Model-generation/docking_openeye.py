"""This module contains functions docking a molecule to a receptor using Openeye.

The code is adapted from this repository: https://github.com/inspiremd/Model-generation
"""
import MDAnalysis as mda
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from functools import cache, partial
from pathlib import Path
from typing import List, Optional
import numpy as np
from openeye import oechem, oedocking, oeomega
import pandas as pd
from mpi4py import MPI
from tqdm import tqdm
from docking_utils import smi_to_structure
from utils import exception_handler
import os

from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

'''
Functions
'''

def smi_to_structure(smiles: str, output_file: Path, forcefield: str = "mmff") -> None:
    """Convert a SMILES file to a structure file.

    Parameters
    ----------
    smiles : str
        Input SMILES string.
    output_file : Path
        EIther an output PDB file or output SDF file.
    forcefield : str, optional
        Forcefield to use for 3D conformation generation
        (either "mmff" or "etkdg"), by default "mmff".
    """
    # Convert SMILES to RDKit molecule object
    mol = Chem.MolFromSmiles(smiles)

    # Add hydrogens to the molecule
    mol = Chem.AddHs(mol)

    # Generate a 3D conformation for the molecule
    if forcefield == "mmff":
        AllChem.EmbedMolecule(mol)
        AllChem.MMFFOptimizeMolecule(mol)
    elif forcefield == "etkdg":
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    else:
        raise ValueError(f"Unknown forcefield: {forcefield}")

    # Write the molecule to a file
    if output_file.suffix == ".pdb":
        writer = Chem.PDBWriter(str(output_file))
    elif output_file.suffix == ".sdf":
        writer = Chem.SDWriter(str(output_file))
    else:
        raise ValueError(f"Invalid output file extension: {output_file}")
    writer.write(mol)
    writer.close()



def from_mol(mol, isomer=True, num_enantiomers=1):
    """
    Generates a set of conformers as an OEMol object
    Inputs:
        mol is an OEMol
        isomers is a boolean controling whether or not the various diasteriomers of a molecule are created
        num_enantiomers is the allowable number of enantiomers. For all, set to -1
    """
    # Turn off the GPU for omega
    omegaOpts = oeomega.OEOmegaOptions()
    omegaOpts.GetTorDriveOptions().SetUseGPU(False)
    omega = oeomega.OEOmega(omegaOpts)

    out_conf = []
    if not isomer:
        ret_code = omega.Build(mol)
        if ret_code == oeomega.OEOmegaReturnCode_Success:
            out_conf.append(mol)
        else:
            oechem.OEThrow.Warning(
                "%s: %s" % (mol.GetTitle(), oeomega.OEGetOmegaError(ret_code))
            )

    elif isomer:
        for enantiomer in oeomega.OEFlipper(mol.GetActive(), 12, True):
            enantiomer = oechem.OEMol(enantiomer)
            ret_code = omega.Build(enantiomer)
            if ret_code == oeomega.OEOmegaReturnCode_Success:
                out_conf.append(enantiomer)
                num_enantiomers -= 1
                if num_enantiomers == 0:
                    break
            else:
                oechem.OEThrow.Warning(
                    "%s: %s" % (mol.GetTitle(), oeomega.OEGetOmegaError(ret_code))
                )
    return out_conf


def from_string(smiles, isomer=True, num_enantiomers=1):
    """
    Generates an set of conformers from a SMILES string
    """
    mol = oechem.OEMol()
    if not oechem.OESmilesToMol(mol, smiles):
        raise ValueError(f"SMILES invalid for string {smiles}")
    else:
        return from_mol(mol, isomer, num_enantiomers)


def from_structure(structure_file: Path) -> oechem.OEMol:
    """
    Generates an set of conformers from a SMILES string
    """
    mol = oechem.OEMol()
    ifs = oechem.oemolistream()
    if not ifs.open(str(structure_file)):
        raise ValueError(f"Could not open structure file: {structure_file}")

    if structure_file.suffix == ".pdb":
        oechem.OEReadPDBFile(ifs, mol)
    elif structure_file.suffix == ".sdf":
        oechem.OEReadMDLFile(ifs, mol)
    else:
        raise ValueError(f"Invalid structure file extension: {structure_file}")

    return mol


def select_enantiomer(mol_list):
    return mol_list[0]


def dock_conf(receptor, mol, max_poses: int = 1):
    dock = oedocking.OEDock()
    dock.Initialize(receptor)
    lig = oechem.OEMol()
    err = dock.DockMultiConformerMolecule(lig, mol, max_poses)
    return dock, lig

# Returns an array of length max_poses from above. This is the range of scores
def ligand_scores(dock, lig):
    return [dock.ScoreLigand(conf) for conf in lig.GetConfs()]


def best_dock_score(dock, lig):
    return ligand_scores(dock, lig)#[0]


def write_ligand(ligand, output_dir: Path, smiles: str, lig_identify: str) -> None:
    # TODO: If MAX_POSES != 1, we should select the top pose to save
    ofs = oechem.oemolostream()
    for it, conf in enumerate(list(ligand.GetConfs())):
        if ofs.open(f'{str(output_dir)}/{lig_identify}/{it}.pdb'):
            oechem.OEWriteMolecule(ofs, conf)
            ofs.close()
    return
    raise ValueError(f"Could not write ligand to {output_path}")


def write_receptor(receptor, output_path: Path) -> None:
    ofs = oechem.oemolostream()
    if ofs.open(str(output_path)):
        mol = oechem.OEMol()
        contents = receptor.GetComponents(mol)#Within
        oechem.OEWriteMolecule(ofs, mol)
        ofs.close()
    return
    raise ValueError(f"Could not write receptor to {output_path}")


#@cache  # Only read the receptor once
def read_receptor(receptor_oedu_file: Path):
    """Read the .oedu file into a GraphMol object."""
    receptor = oechem.OEDesignUnit()
    oechem.OEReadDesignUnit(str(receptor_oedu_file), receptor)
    return receptor

#@cache
def create_proteinuniv(protein_pdb):
    protein_universe = mda.Universe(protein_pdb)
    return protein_universe

def create_complex(protein_universe, ligand_pdb):
    u1 = protein_universe
    u2 = mda.Universe(ligand_pdb)
    u = mda.core.universe.Merge(u1.select_atoms("all"), u2.atoms)#, u3.atoms)
    return u

def create_trajectory(protein_universe, ligand_dir, output_pdb_name, output_dcd_name):
    import MDAnalysis as mda
    ligand_files = sorted(os.listdir(ligand_dir))
    comb_univ_1 = create_complex(protein_universe, f'{ligand_dir}/{ligand_files[0]}').select_atoms("all")

    with mda.Writer(output_pdb_name, comb_univ_1.n_atoms) as w:
        w.write(comb_univ_1)
    with mda.Writer(output_dcd_name, comb_univ_1.n_atoms,) as w:
        for it, ligand_file in enumerate(ligand_files):
            comb_univ = create_complex(protein_universe, f'{ligand_dir}/{ligand_file}') 
            w.write(comb_univ)    # write a whole universe
            os.remove(f'{ligand_dir}/{ligand_file}')
    return

@exception_handler(default_return=0.0)
def run_docking(
    smiles: str, 
    lig_identify: str,
    receptor_oedu_file: Path,
    max_confs: int,
    score_cutoff: float,
    protein_pdb: Path,
    temp_dir: Path,
    out_lig_dir: Path,
    temp_storage: Optional[Path] = None
) -> float:
    """Run OpenEye docking on a single ligand.

    Parameters
    ----------
    smiles : ste
        A single SMILES string.
    receptor_oedu_file : Path
        Path to the receptor .oedu file.
    max_confs : int
        Number of ligand poses to generate
    temp_lig_dir : Path
        Temporary directory to write individual ligand poses
    out_lig_path : Path
        Location to output ligand_protein top n poses --> single pdb
    out_rec_path : Path
        Where to write output receptor file
    temp_storage : Path
        Path to the temporary storage directory to write structures to,
        if None, use the current working Python's built in temp storage.

    Returns
    -------
    float
        The docking score of the best conformer.
    """

    try:
        conformers = select_enantiomer(from_string(smiles))
    except:
        with tempfile.NamedTemporaryFile(suffix=".pdb", dir=temp_storage) as fd:
            # Read input SMILES and generate conformer
            smi_to_structure(smiles, Path(fd.name))
            conformers = from_structure(Path(fd.name))

    # Read the receptor to dock to
    receptor = read_receptor(receptor_oedu_file)

    # Dock the ligand conformers to the receptor
    dock, lig = dock_conf(receptor, conformers, max_poses=max_confs)

    # Get the docking scores
    best_score = best_dock_score(dock, lig)

    # If receptor pdb file doesnt exist then create one
    if True:
        protein_pdb = f'{temp_dir}/{Path(receptor_oedu_file).stem}.pdb'
        if not os.path.isfile(protein_pdb):
            write_receptor(receptor, protein_pdb)

    if best_score[0]<score_cutoff:
        sys.stdout.flush()
        if not os.path.isdir(f'{temp_dir}/{lig_identify}'):
            os.mkdir(f'{temp_dir}/{lig_identify}')
        write_ligand(lig, temp_dir, smiles, lig_identify)
        protein_universe = create_proteinuniv(protein_pdb)
        create_trajectory(protein_universe, f'{temp_dir}/{lig_identify}', f'{out_lig_dir}/pdbs/{Path(receptor_oedu_file).stem}.{lig_identify}.pdb',f'{out_lig_dir}/dcds/{Path(receptor_oedu_file).stem}.{lig_identify}.dcd')
    return np.mean(best_score)

def run_mpi_docking(
    rank,
    size,
    smiles_df,
    receptor_oedu_file: Path,
    max_confs: int,
    score_cutoff: float,
    protein_pdb: Path,
    temp_dir: Path,
    out_lig_dir: Path,
    out_csv_pattern: str,
    temp_storage: Optional[Path] = None) -> float:
    
    smiles_list = smiles_df['smiles'].tolist()
    smiles_index = [ind for ind in range(len(smiles_list))]
    smiles_list_new = np.array_split(smiles_list, size)[rank]
    smiles_index_new = np.array_split(smiles_index, size)[rank]
    scores = [run_docking(smi, f'{smiles_index_new[it]}', receptor_oedu_file, max_confs, score_cutoff, protein_pdb, temp_dir, out_lig_dir) for it, smi in enumerate(smiles_list_new)]
    df_new = pd.DataFrame({'Index':smiles_index_new, 'SMILES':smiles_list_new, 'Scores':scores})
    df_new.to_csv(f'{out_csv_pattern}.{rank}.csv', index=False)
    return 1

'''
Running Code
'''

### Initialize mpi
comm = MPI.COMM_WORLD
rank = comm.Get_rank() 
size = comm.Get_size()

### Parameters setting

df_smiles = pd.read_csv('input/RTCB_TopHits_22B_100k.csv')[:1000] # smilesdatafile. smiles column is 'smiles'
recep_file = 'input/rtcb-7p3b-receptor-5GP-A-DU-601.oedu' # receptor oedu file for openeye 
max_confs = 10 # confs to generate
score_cutoff = -10 # below this score generate pose
temp_dir = 'lig_confs' # store ligand poses temporarily
output_traj = 'output_combined_trajectories' # store pdb + dcd files for complex poses
score_pattern = 'scores/rtcb_scores' #store scores like this (will have #ranks files)
protein_pdb = 'prot.pdb' #protein pdb file to use to store. Will save everything in this file to complex


### Running docking in parallel
run_mpi_docking(rank, size, df_smiles, recep_file, max_confs, score_cutoff, protein_pdb, temp_dir, output_traj, score_pattern, 'temp')


'''
combining all score files into one. For now just do this outside of the script
'''
if False:
    comm.Barrier()
    if rank==0:
        df = pd.read_csv(f'{score_pattern}.0.csv')
        for r in range(1, size):
            df = pd.concat([df, pd.read_csv(f'{score_pattern}.{r}.csv')])
        df.to_csv(f'{score_pattern}.csv')

