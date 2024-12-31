# type: ignore
from copy import deepcopy

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from e3nn.util.jit import prepare
import ase.build
import ase.io
from torch_geometric.transforms.pad import Pad, AttrNamePadding

import sevenn.util as util
from sevenn.train.graph_dataset import SevenNetGraphDataset
from sevenn.model_build import build_E3_equivariant_model
from sevenn._const import DEFAULT_E3_EQUIVARIANT_MODEL_CONFIG, NUM_UNIV_ELEMENT


device = torch.device('cuda')
torch._dynamo.config.capture_scalar_outputs = True


model_conf = deepcopy(DEFAULT_E3_EQUIVARIANT_MODEL_CONFIG)
ntype = NUM_UNIV_ELEMENT
model_conf.update(
    {
        'shift': [1.0] * (NUM_UNIV_ELEMENT),
        'scale': [1.0] * (NUM_UNIV_ELEMENT),
        'conv_denominator': 10.0,
    }
)
model_conf.update({'lmax': 2, 'is_parity': False})
model_conf.update(**util.chemical_species_preprocess([], True))

atoms = ase.build.bulk('NaCl', 'rocksalt', a=4.00, cubic=True) * (3, 3, 3)
atoms.rattle(100.0)
atoms.info['y_energy'] = 1.0
atoms.arrays['y_force'] = np.random.rand(len(atoms), 3)
ase.io.write('tmp.extxyz', atoms)

atoms = ase.build.bulk('NaCl', 'rocksalt', a=4.00, cubic=True) * (3, 3, 3)
atoms.rattle(10.0)
atoms.info['y_energy'] = 1.0
atoms.arrays['y_force'] = np.random.rand(len(atoms), 3)
ase.io.write('tmp2.extxyz', atoms)

gd = SevenNetGraphDataset(
    cutoff=model_conf.get('cutoff', 4.5),
    root='./',
    files=['tmp.extxyz', 'tmp2.extxyz'],
    energy_key='y_energy',
    force_key='y_force',
    processed_name='dynamic',
)

loader = DataLoader(gd, batch_size=1)

model = build_E3_equivariant_model(model_conf)
model = model.to(device)
model.set_is_batch_data(False)

for system in loader:
    system = system.to(device)
    out = model(system)
    print(out['inferred_total_energy'])
    print(out['inferred_force'].max().item())


max_num_nodes = max(gd[0]['num_atoms'].item(), gd[1]['num_atoms'].item()) + 10
max_num_edges = (
    max(gd[0]['edge_index'].shape[1], gd[1]['edge_index'].shape[1]) + 10000
)

pad = model.prepare_padding(max_num_nodes, max_num_edges)

for system in loader:
    system = system.to(device)
    system = pad(system)
    out = model(system)
    print(out['inferred_total_energy'])
    print(out['inferred_force'].max().item())


torch._dynamo.reset()
model_prep = prepare(build_E3_equivariant_model, allow_autograd=True)(model_conf)  # type: ignore
pad = model_prep.prepare_padding(max_num_nodes, max_num_edges)
model_prep.load_state_dict(model.state_dict(), strict=False)
model_prep = model_prep.to(device)

_seq = torch.compile(
    model_prep._seq, mode='max-autotune', fullgraph=True, dynamic=False
)

model_prep.set_is_batch_data(False)

for batch in loader:
    batch = batch.to(device)
    batch = pad(batch)
    batch = model_prep._preprocess(batch)
    out2 = _seq(batch)
    print(out2['inferred_total_energy'])
    print(out2['inferred_force'].max().item())


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
        batch = pad(batch)
        batch = model_prep._preprocess(batch)
        out2 = _seq(batch)
end.record()
torch.cuda.synchronize()
print('compile:')
print(start.elapsed_time(end))
