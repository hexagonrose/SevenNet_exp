import torch
from torch_geometric.loader import DataLoader

from e3nn.util.jit import prepare

import sevenn.util as util
from sevenn.train.trainer import Trainer
from sevenn.train.graph_dataset import SevenNetGraphDataset
from sevenn.model_build import build_E3_equivariant_model


device = torch.device('cuda')
torch._dynamo.config.capture_scalar_outputs = True
torch.set_float32_matmul_precision('highest')

model, config = util.model_from_checkpoint('./checkpoint_200.pth')

gd = SevenNetGraphDataset(
    cutoff=config.get('cutoff', 4.5),
    root='/home/parkyutack/dataset/',
    processed_name='MPF_test.pt',
    force_reload=False,
)

loader = DataLoader(gd, batch_size=2)

model.train()
model = model.to(device)
model.set_is_batch_data(True)

torch._dynamo.reset()
model_opt = prepare(build_E3_equivariant_model, allow_autograd=True)(config)  # type: ignore
model_opt.train()  # must be turned on before compile, if gonna use with trainer?
model_opt.load_state_dict(model.state_dict())
model_opt = model_opt.to(device)


def train_step(batch, model, optimizer, loss_functions):
    optimizer.zero_grad()
    output = model._seq(batch)
    total_loss = torch.tensor([0.0], device=device)
    for loss_def, w in loss_functions:
        indv_loss = loss_def.get_loss(output, model)
        if indv_loss is not None:
            total_loss += indv_loss * w
    total_loss.backward()
    optimizer.step()


args, _, _ = Trainer.args_from_checkpoint('./checkpoint_200.pth')
args['model'] = model_opt
# args['model'] = model
trainer = Trainer(**args)  # for help, not used
torch._dynamo.reset()
train_step_opt = torch.compile(train_step, mode='default')

for _ in range(10):
    for batch in loader:
        trainer.model._preprocess(batch)
        batch = batch.to(device, non_blocking=True)
        batch = batch.to_dict()
        train_step_opt(
            batch, trainer.model, trainer.optimizer, trainer.loss_functions
        )


from torch_geometric.transforms.pad import Pad, AttrNamePadding

for batch in loader:
    batch.to(device)
    del batch.cell_lattice_vectors
    del batch.pbc_shift
    n_node = batch.num_atoms
    break

ref = model(batch.clone())

pad_node = AttrNamePadding(
    {
        'batch': torch.max(batch['batch']).item() + 1,
    }
)
pd = Pad(16 * 2 + 1, 1080 + 10, node_pad_value=pad_node, add_pad_mask=True)
# pd = Pad(16*2 + 1, 1080 + 10, add_pad_mask=True)

pbatch = pd(batch)

pbatch = model._preprocess(pbatch)
pbatch['_s'] = torch.zeros(len(pbatch['atomic_numbers']), 6)
pbatch['num_atoms'][0] = 16 + 1
pbatch['cell_volume'] = torch.cat([pbatch['cell_volume'], torch.zeros(1, device=device)])

pbatch_original = pbatch.clone()


cnt = 0
stop = 26
pbatch_x = pbatch.clone()
pbatch_x = pbatch_x.to(device)
for name, mm in model._modules.items():
    print(name)
    pbatch_x = mm(pbatch_x)
    cnt += 1
    if cnt == stop:
        break

