# type: ignore
from copy import deepcopy

from tqdm import tqdm
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from e3nn.util.jit import prepare
import ase.build
import ase.io

import sevenn.util as util
from sevenn.train.graph_dataset import SevenNetGraphDataset
from sevenn.model_build import build_E3_equivariant_model


device = torch.device('cuda')
torch._dynamo.config.capture_scalar_outputs = True

cp = util.load_checkpoint('7net-0')
model_conf = cp.config

atoms = ase.build.bulk('Si').repeat(8)
atoms.info['y_energy'] = 1.0
atoms.arrays['y_force'] = np.random.rand(len(atoms), 3)
ase.io.write('tmp.extxyz', atoms)

gd = SevenNetGraphDataset(
    cutoff=model_conf.get('cutoff', 5.0),
    root='./',
    files=['tmp.extxyz'] * 2,
    energy_key='y_energy',
    force_key='y_force',
    force_reload=True,
)

loader = DataLoader(gd, batch_size=1)

model = cp.build_model()
assert not isinstance(model, list)
model = model.to(device)
model.set_is_batch_data(True)

for batch in loader:
    batch = batch.to(device)
    out = model(batch)
    break
print(out['inferred_total_energy'])

torch._dynamo.reset()
model_prep = cp.build_model(compile=True)
model_prep.to('cuda')

for batch in loader:
    batch = batch.to(device)
    out2 = model_prep(batch)
    break
print(out2['inferred_total_energy'])


model.eval()
# bench original
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
for _ in tqdm(range(25)):
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
for _ in tqdm(range(25)):
    for batch in loader:
        batch = batch.to(device)
        out2 = model_prep(batch)
end.record()
torch.cuda.synchronize()
print('compile:')
print(start.elapsed_time(end))

