import pandas as pd
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

### Initialize mpi
if False:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank() 
    size = comm.Get_size()
    print(rank)
    print(size)

# Function to add backbone position restraints
def add_backbone_posres(system, positions, atoms, restraint_force):
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

################################################
################################################


mdsteps = 50000000
node_val = 0
if True:
    from openff.toolkit import Molecule
    
    df = pd.read_csv('smiles_to_param.smi')
    df_smiles = df['smiles'].tolist()
    df_id = df['id'].tolist()
    smi = df_smiles[node_val]
    id_val = df_id[node_val]
    print(smi)
    print(id_val)
    
    molecule = Molecule.from_smiles(smi)
    # Create the GAFF template generator
    from openmmforcefields.generators import GAFFTemplateGenerator
    gaff = GAFFTemplateGenerator(molecules=molecule)
    
    # Create an OpenMM ForceField object with AMBER ff14SB and TIP3P with compatible ions
    from openmm.app import ForceField
    forcefield = ForceField('amber/protein.ff14SB.xml', 'amber/tip3p_standard.xml', 'amber/tip3p_HFE_multivalent.xml')
    # Register the GAFF template generator
    forcefield.registerTemplateGenerator(gaff.generator)
    
    
    # You can now parameterize an OpenMM Topology object that contains the specified molecule.
    # forcefield will load the appropriate GAFF parameters when needed, and antechamber
    # will be used to generate small molecule parameters on the fly.
    from openmm.app import PDBFile
    modeller = PDBFile(f'output_dcds/output.{id_val}.pdb')
    system = forcefield.createSystem(modeller.topology, 
                                    nonbondedMethod=app.PME,
                                    removeCMMotion=True,
                                    nonbondedCutoff=12.0*u.angstroms,
                                    constraints=app.HBonds,
                                    switchDistance=10.0*u.angstroms)#,
                                    #hydrogenMass=4*amu
                                    #)
    
    print(system)
    barostat = system.addForce(MonteCarloBarostat(1*atmosphere, 310*kelvin))
    
    integrator = LangevinIntegrator(310*kelvin, 1/picosecond, 0.002*picoseconds)
    integrator.setConstraintTolerance(0.00001)
    platform = Platform.getPlatformByName('CUDA')
    
    #localrank = rank%4
    properties = {'DeviceIndex': f'{0,1,2,3}', 'Precision': 'mixed'}
    
    #simulation = Simulation(psf.topology, posres_sys, integrator, platform, properties)
    simulation = Simulation(modeller.topology, system, integrator, platform, properties)
    simulation.loadCheckpoint(f'output_dcds/eq.{id_val}.chk')

    eq_state = simulation.context.getState(getVelocities=True, getPositions=True)
    positions = eq_state.getPositions()
    velocities = eq_state.getVelocities()
    
    simulation.context.setPositions(positions)
    simulation.context.setVelocities(velocities)

    ### DCD reporter
    simulation.reporters.append(
        DCDReporter(f'output_dcds/prod.0.{id_val}.dcd', 5000))
    
    ### Data reporter
    simulation.reporters.append(
        StateDataReporter(
             f'prod.0.{id_val}.csv',
             10000,
             step=True,
             potentialEnergy=True,
             temperature=True,
             progress=True,
             remainingTime=True,
            speed=True,
            volume=True,
            totalSteps=mdsteps,
            separator='\t'
            )
        )
    
    ### Checkpointer
    simulation.reporters.append(
        CheckpointReporter(
            f'output_dcds/prod.0.{id_val}.restart.chk',
            100000
            )
        )
    
    ############# Run simulation! #############
    
    print('Running Production...')
    simulation.step(mdsteps)
    simulation.saveState('output_dcds/prod.0.{id_val}.state')
    simulation.saveCheckpoint('output/prod.0.{id_val}.chk')









































