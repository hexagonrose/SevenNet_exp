# type: ignore
import sys

import torch

import numpy as np
from ase.md import MDLogger
from ase.io.trajectory import Trajectory
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.build import bulk
from ase import units

from sevenn.sevennet_calculator import SevenNetCalculator

torch._inductor.config.compile_threads = 8

def print_md_log(dyn=..., atoms=...):
    step = dyn.get_number_of_steps()
    time_fs = step * dyn.dt / (1.0 * units.fs)  # if dt in ASE units (e.g., 1 fs)
    nedges = atoms.calc.results['num_edges']
    temp = atoms.get_temperature()
    epot = atoms.get_potential_energy()
    print(f'{step},{time_fs:.1f},{temp:.3f},{epot:.3f},{nedges}', flush=True)


calc = SevenNetCalculator('7net-0', device='cuda', compile=True, compile_kwargs={'mode': 'reduce-overhead'})

atoms = bulk('Si').repeat(14)
atoms.calc = calc

timestep = 1.0 * units.fs
dyn = VelocityVerlet(atoms, timestep=timestep)

temperature = 300  # Kelvin
MaxwellBoltzmannDistribution(
    atoms, temperature_K=temperature, rng=np.random.RandomState(7)
)

print(atoms.get_potential_energy())
dyn.run(1)
print(atoms.get_potential_energy())
# traj = Trajectory('md_output.traj', 'w', atoms)
# logger = MDLogger(dyn, atoms, sys.stdout, header=True, stress=False, peratom=True)

# dyn.attach(logger, interval=10)  # Print progress every 10 steps
# dyn.attach(traj.write, interval=10)  # Write trajectory every 10 steps
dyn.attach(print_md_log, interval=1, dyn=dyn, atoms=atoms)


print('step,time,temp (K),E_pot (eV),nedges')
# bench compiled
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
dyn.run(10)
end.record()
torch.cuda.synchronize()
print('Wall time:')
print(start.elapsed_time(end))
