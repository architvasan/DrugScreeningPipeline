from mpi4py import MPI
import MDAnalysis as mda
import time
from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout, exit, stderr
from parmed import unit as u
from copy import deepcopy
import sys
from sys import stdout
from openff.toolkit import Molecule
from openmmforcefields.generators import GAFFTemplateGenerator
import pandas as pd
import numpy as np

'''
Fix input prot_lig complex pdb file
'''
def fix_prot_ligpdb(fixed_dir, input_pdb):
    u = mda.Universe(input_pdb)
    #sel = u.select_atoms('(chainID A or resname UNL) and (not resname 5GP) and not (resname HOH and same residue as around 1 resname UNL)')
    sel = u.select_atoms('(chainID A or resname UNL) and (not resname 5GP) and not (resname HOH)')

    input_base = os.path.basename(input_pdb).split('.')
    print(input_base)
    sel.write(f'{fixed_dir}/{input_base[0]}.{input_base[1]}.pdb')
    return f'{fixed_dir}/{input_base[0]}.{input_base[1]}.pdb'

'''
Create Amber forcefield parameters 
    protein ff14SB,
    TIP3P with compatible ions,
    and add GAFF for molecules
'''
def create_forcefield(smi):
    molecule = Molecule.from_smiles(smi)
    # Create the GAFF template generator
    gaff = GAFFTemplateGenerator(molecules=molecule)
    forcefield = ForceField('amber/protein.ff14SB.xml',
                            'amber/tip3p_standard.xml',
                            'amber/tip3p_HFE_multivalent.xml')
    # Register the GAFF template generator
    forcefield.registerTemplateGenerator(gaff.generator)
    return forcefield

'''
Create system with 
    water,
    ions,
    protein,
    and compound
'''
def create_system(fixedpdb,
                forcefield):

    pdbfile = PDBFile(fixedpdb)
    modeller = Modeller(pdbfile.topology, pdbfile.positions)
    
    modeller.addSolvent(forcefield,
                        model='tip3p',
                        padding=1*nanometer,
                        ionicStrength=0.15*molar)

    system = forcefield.createSystem(modeller.topology, 
                                    nonbondedMethod=app.PME,
                                    removeCMMotion=True,
                                    nonbondedCutoff=12.0*u.angstroms,
                                    constraints=app.HBonds,
                                    switchDistance=10.0*u.angstroms,
                                    hydrogenMass=4*amu
                                    )
    return system, modeller

'''
Function to add backbone position restraints
'''
def add_backbone_posres(system, 
                        positions,
                        atoms,
                        restraint_force):

  force = CustomExternalForce("k*periodicdistance(x, y, z, x0, y0, z0)^2")
  force_amount = restraint_force * kilocalories_per_mole/angstroms**2
  force.addGlobalParameter("k", force_amount)
  force.addPerParticleParameter("x0")
  force.addPerParticleParameter("y0")
  force.addPerParticleParameter("z0")
  for i, (atom_crd, atom) in enumerate(zip(positions, atoms)):
    if atom.name in  ('CA', 'C', 'N', 'O'):
      force.addParticle(i, atom_crd.value_in_unit(nanometers))
  posres_sys = deepcopy(system)
  posres_sys.addForce(force)
  return posres_sys

'''
Set up simulation
'''
def setup_sim(localrank,
            modeller,
            posres_sys):

    integrator = LangevinIntegrator(310*kelvin,
                                    1/picosecond,
                                    0.004*picoseconds)

    integrator.setConstraintTolerance(0.00001)
    platform = Platform.getPlatformByName('CUDA')
    properties = {'DeviceIndex': f'{0}', 'Precision': 'mixed'}
    
    simulation = Simulation(modeller.topology,
                            posres_sys,
                            integrator)
                            #,
                            #properties)

    simulation.context.setPositions(modeller.positions)
    simulation.reporters.append(
        StateDataReporter(
            'equilibrate.log',
            5000,
            step=True,
            potentialEnergy=True,
            temperature=True,
            volume=True,
            density=True
        )
    )
    return simulation

'''
Run minimization + equilibration
'''
def min_equil(simulation,
            outpatt,
            out_dcd_step,
            numsteps):

    #Minimize
    print('Minimizing...')
    simulation.minimizeEnergy()
    
    positions = simulation.context.getState(getPositions=True).getPositions()
    PDBFile.writeFile(simulation.topology,
                        positions,
                        open(f'{outpatt}.pdb', 'w'))
    
    simulation.reporters.append(DCDReporter(f'{outpatt}.dcd',
                                out_dcd_step))
     
    simulation.reporters.append(StateDataReporter(stdout,
                                out_dcd_step,
                                step=True,
                                potentialEnergy=True,
                                temperature=True,
                                progress=True,
                                remainingTime=True,
                                speed=True,
                                volume=True,
                                totalSteps=numsteps))
    print("Running Simulations") 
    simulation.step(numsteps)
    return None

'''
Stitch all functions together
'''
def run_sim(smi,
            input_pdb,
            fixed_dir,
            restraint_force,
            localrank,
            outpatt,
            out_dcd_step,
            numsteps):

    fixedpdb = fix_prot_ligpdb(fixed_dir, input_pdb)
    print(f"Fixed pdb: rank {localrank}")
    forcefield = create_forcefield(smi)
    print(f"Created forcefield: rank {localrank}")
    system, modeller = create_system(fixedpdb,
                                    forcefield)
    print(f"Created system with wwater + ions: rank {localrank}")

    posres_sys = add_backbone_posres(system, 
                        modeller.positions,
                        modeller.topology.atoms(),
                        restraint_force)
    print(f"added backbone restraints: rank {localrank}")

    simulation = setup_sim(localrank,
                            modeller,
                            posres_sys)

    print(f"set up sim: rank {localrank}")
    stdout.flush()

    if False:
        min_equil(simulation,
                    outpatt,
                    out_dcd_step,
                    numsteps)

    return None

'''
Run simulations in parallel
'''
def run_sim_mpi(rank,
                size,
                ppn,
                input_smi_file,
                inpdb_dir,
                inpdb_patt,
                fixed_dir,
                restraint_force,
                outdcd_dir,
                out_dcd_step,
                numsteps) -> None:

    localrank = int(rank%ppn)+1
    smiles_df = pd.read_csv(input_smi_file)
    list_smiles = smiles_df['SMILES'].tolist()
    list_index = smiles_df['Index'].tolist()

    list_smiles = np.array_split(list_smiles, size)[rank]
    list_index = np.array_split(list_index, size)[rank]
    
    for smi, ind in zip(list_smiles, list_index):
        input_pdb = f'{inpdb_dir}/{inpdb_patt}.{ind}.pdb'
        outpatt = f'{outdcd_dir}/{inpdb_patt}.{ind}'
        try:
            run_sim(smi,
                input_pdb,
                fixed_dir,
                restraint_force,
                localrank,
                outpatt,
                out_dcd_step,
                numsteps)
        except:
            print(f"error on {smi}")
            continue
########################################################
############## Run mpi_sim for input file ##############
########################################################

### Initialize mpi
comm = MPI.COMM_WORLD
rank = comm.Get_rank() 
size = comm.Get_size()
ppn = 30

input_smi_file = 'input_smi_fordocking.csv' 
inpdb_dir = '../Model-generation/output_combined_trajectories/pdbs'
inpdb_patt = 'rtcb-7p3b-receptor-5GP-A-DU-601'
fixed_dir = 'fixed_pdbs' 
restraint_force = 10
outdcd_dir = 'output_dcds'
out_dcd_step = 1000
numsteps = 250000

run_sim_mpi(rank,
            size,
            ppn,
            input_smi_file,
            inpdb_dir,
            inpdb_patt,
            fixed_dir,
            restraint_force,
            outdcd_dir,
            out_dcd_step,
            numsteps)


################################################
################################################
if False:
    #posres_sys = add_backbone_posres(system, modeller.positions, modeller.topology.atoms(), 10)
    #posres_sys = add_backbone_posres(system, pdbfile.positions, psf.topology.atoms(), 42)
    integrator = LangevinIntegrator(310*kelvin, 1/picosecond, 0.004*picoseconds)
    integrator.setConstraintTolerance(0.00001)
    platform = Platform.getPlatformByName('CUDA')
    properties = {'DeviceIndex': f'0', 'Precision': 'mixed'}
    
    #simulation = Simulation(psf.topology, posres_sys, integrator, platform, properties)
    simulation = Simulation(modeller.topology, posres_sys, integrator, properties)
    simulation.context.setPositions(modeller.positions)
    simulation.reporters.append(
        StateDataReporter(
            'equilibrate.log',
            5000,
            step=True,
            potentialEnergy=True,
            temperature=True,
            volume=True,
            density=True
        )
    )
    
    
    #sys.exit()
    
    # Set up
    # Load structure/psf
    
