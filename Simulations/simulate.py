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

if False:
    u = mda.Universe('rtcb-7p3b-receptor-5GP-A-DU-601.453_new2.pdb')
    sel = u.select_atoms('not (resname HOH and same residue as around 1 resname UNL)')
    sel.write('rtcb-rtcb-receptor-5GP-A-DU-601.453_new3.pdb')

from openff.toolkit import Molecule
molecule = Molecule.from_smiles('Cc1ccc2nc(COc3cc(Br)ccc3C(=O)NO)cc(=O)n2c1')
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
pdbfile = PDBFile('rtcb-7p3b-receptor-5GP-A-DU-601.453.pdb')
modeller = Modeller(pdbfile.topology, pdbfile.positions)

modeller.addSolvent(forcefield, model='tip3p', padding=1*nanometer, ionicStrength=0.15*molar)
system = forcefield.createSystem(modeller.topology, 
                                nonbondedMethod=app.PME,
                                removeCMMotion=True,
                                nonbondedCutoff=12.0*u.angstroms,
                                constraints=app.HBonds,
                                switchDistance=10.0*u.angstroms,
                                hydrogenMass=4*amu
                                )

print(system)

posres_sys = add_backbone_posres(system, modeller.positions, modeller.topology.atoms(), 10)
#posres_sys = add_backbone_posres(system, pdbfile.positions, psf.topology.atoms(), 42)
integrator = LangevinIntegrator(310*kelvin, 1/picosecond, 0.004*picoseconds)
integrator.setConstraintTolerance(0.00001)
platform = Platform.getPlatformByName('CUDA')
properties = {'DeviceIndex': f'0', 'Precision': 'mixed'}

#simulation = Simulation(psf.topology, posres_sys, integrator, platform, properties)
simulation = Simulation(modeller.topology, posres_sys, integrator)
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

#Minimize
print('Minimizing...')
#simulation.context.setPositions(modeller.positions)
simulation.minimizeEnergy()

simulation.reporters.append(PDBReporter('output.pdb', 1000))
simulation.reporters.append(StateDataReporter(stdout, 1000, step=True,
        potentialEnergy=True, temperature=True,
         progress=True,
         remainingTime=True,
        speed=True,
        volume=True,
        totalSteps=10000))



simulation.step(10000)

#sys.exit()

# Set up
# Load structure/psf

