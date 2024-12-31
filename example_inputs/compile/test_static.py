# type: ignore
from copy import deepcopy

import numpy as np

import torch
from torch_geometric.loader import DataLoader

from e3nn.util.jit import prepare

import ase.build
import ase.io

import sevenn.util as util
from sevenn.train.graph_dataset import SevenNetGraphDataset
from sevenn.model_build import build_E3_equivariant_model
from sevenn._const import DEFAULT_E3_EQUIVARIANT_MODEL_CONFIG


device = torch.device('cuda')
torch._dynamo.config.capture_scalar_outputs = True


model_conf = deepcopy(DEFAULT_E3_EQUIVARIANT_MODEL_CONFIG)
model_conf.update({'shift': 1.0, 'scale': 1.0, 'conv_denominator': 10.0})
model_conf.update({'lmax': 2, 'is_parity': False})
model_conf.update(**util.chemical_species_preprocess([], True))

atoms = ase.build.bulk('NaCl', 'rocksalt', a=4.00, cubic=True) * (3, 3, 3)
atoms.info['y_energy'] = 1.0
atoms.arrays['y_force'] = np.random.rand(len(atoms), 3)
ase.io.write('tmp.extxyz', atoms)

gd = SevenNetGraphDataset(
    cutoff=model_conf.get('cutoff', 4.5),
    root='./',
    files=['tmp.extxyz'] * 2,
    energy_key='y_energy',
    force_key='y_force',
)

loader = DataLoader(gd, batch_size=2)

model = build_E3_equivariant_model(model_conf)
assert not isinstance(model, list)
model = model.to(device)
model.set_is_batch_data(True)

for batch in loader:
    batch = batch.to(device)
    out = model(batch)
    break
print(out['inferred_total_energy'])

torch._dynamo.reset()
model_prep = prepare(build_E3_equivariant_model, allow_autograd=True)(model_conf)  # type: ignore
model_prep.load_state_dict(model.state_dict())
model_prep = model_prep.to(device)

_seq = torch.compile(model_prep._seq, mode='max-autotune', fullgraph=True)

"""
torch._dynamo.reset()
for batch in loader:
    batch = batch.to(device)
    exp = torch._dynamo.explain(model_comp, batch)
    break
"""

for batch in loader:
    batch = batch.to(device)
    batch = model_prep._preprocess(batch)
    out2 = _seq(batch)
    break
print(out2['inferred_total_energy'])


# bench original
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(50):
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
end.record()
torch.cuda.synchronize()
print('original:')
print(start.elapsed_time(end))


# bench compiled
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(50):
    for batch in loader:
        batch = batch.to(device)
        batch = model_prep._preprocess(batch)
        out2 = _seq(batch)
end.record()
torch.cuda.synchronize()
print('compile:')
print(start.elapsed_time(end))

