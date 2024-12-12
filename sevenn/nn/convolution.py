from itertools import chain
from typing import List

import torch
import torch.nn as nn
from e3nn.nn import FullyConnectedNet
from e3nn.o3 import Irreps, TensorProduct, wigner_3j
from e3nn.util.jit import compile_mode

import sevenn._keys as KEY
from sevenn._const import AtomGraphDataType

from .activation import ShiftedSoftPlus
from .util import broadcast


def message_gather(
    node_features: torch.Tensor, edge_dst: torch.Tensor, message: torch.Tensor
):
    index = broadcast(edge_dst, message, 0)
    out_shape = [len(node_features)] + list(message.shape[1:])
    out = torch.zeros(
        out_shape, dtype=node_features.dtype, device=node_features.device
    )
    out.scatter_reduce_(0, index, message, reduce='sum')
    return out


@compile_mode('script')
class IrrepsConvolution(nn.Module):
    """
    convolution of (fig 2.b), comm. in LAMMPS
    """

    def __init__(
        self,
        irreps_x: Irreps,
        irreps_filter: Irreps,
        irreps_out: Irreps,
        weight_layer_input_to_hidden: List[int],
        weight_layer_act=ShiftedSoftPlus,
        denominator: float = 1.0,
        train_denominator: bool = False,
        data_key_x: str = KEY.NODE_FEATURE,
        data_key_filter: str = KEY.EDGE_ATTR,
        data_key_weight_input: str = KEY.EDGE_EMBEDDING,
        data_key_edge_idx: str = KEY.EDGE_IDX,
        lazy_layer_instantiate: bool = True,
        is_parallel: bool = False,
    ):
        super().__init__()
        self.denominator = nn.Parameter(
            torch.FloatTensor([denominator]), requires_grad=train_denominator
        )
        self.key_x = data_key_x
        self.key_filter = data_key_filter
        self.key_weight_input = data_key_weight_input
        self.key_edge_idx = data_key_edge_idx
        self.is_parallel = is_parallel

        instructions = []
        irreps_mid = []
        weight_numel = 0
        for i, (mul_x, ir_x) in enumerate(irreps_x):
            for j, (_, ir_filter) in enumerate(irreps_filter):
                for ir_out in ir_x * ir_filter:
                    if ir_out in irreps_out:  # here we drop l > lmax
                        k = len(irreps_mid)
                        weight_numel += mul_x * 1  # path shape
                        irreps_mid.append((mul_x, ir_out))
                        instructions.append((i, j, k, 'uvu', True))

        irreps_mid = Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort()  # type: ignore
        instructions = [
            (i_in1, i_in2, p[i_out], mode, train)
            for i_in1, i_in2, i_out, mode, train in instructions
        ]

        # From v0.11.x, to compatible with cuEquivariance
        self._instructions_before_sort = instructions
        instructions = sorted(instructions, key=lambda x: x[2])

        self.convolution_kwargs = dict(
            irreps_in1=irreps_x,
            irreps_in2=irreps_filter,
            irreps_out=irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )

        self.weight_nn_kwargs = dict(
            hs=weight_layer_input_to_hidden + [weight_numel], act=weight_layer_act
        )

        self.convolution = None
        self.weight_nn = None
        self.layer_instantiated = False
        self.convolution_cls = TensorProduct
        self.weight_nn_cls = FullyConnectedNet

        if not lazy_layer_instantiate:
            self.instantiate()

        self._comm_size = irreps_x.dim  # used in parallel

    def instantiate(self):
        if self.convolution is not None:
            raise ValueError('Convolution layer already exists')
        if self.weight_nn is not None:
            raise ValueError('Weight_nn layer already exists')

        self.convolution = self.convolution_cls(**self.convolution_kwargs)
        self.weight_nn = self.weight_nn_cls(**self.weight_nn_kwargs)
        self.layer_instantiated = True

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        assert self.convolution is not None, 'Convolution is not instantiated'
        assert self.weight_nn is not None, 'Weight_nn is not instantiated'
        weight = self.weight_nn(data[self.key_weight_input])
        x = data[self.key_x]
        if self.is_parallel:
            x = torch.cat([x, data[KEY.NODE_FEATURE_GHOST]])

        # note that 1 -> src 0 -> dst
        edge_src = data[self.key_edge_idx][1]
        edge_dst = data[self.key_edge_idx][0]

        message = self.convolution(x[edge_src], data[self.key_filter], weight)

        x = message_gather(x, edge_dst, message)
        x = x.div(self.denominator)
        if self.is_parallel:
            x = torch.tensor_split(x, data[KEY.NLOCAL])[0]
        data[self.key_x] = x
        return data


@compile_mode('script')
class CGAfterGatherConvolution(nn.Module):
    def __init__(
        self,
        irreps_x: Irreps,
        irreps_filter: Irreps,
        irreps_out: Irreps,
        weight_layer_input_to_hidden: List[int],
        weight_layer_act=ShiftedSoftPlus,
        denominator: float = 1.0,
        train_denominator: bool = False,
        data_key_x: str = KEY.NODE_FEATURE,
        data_key_filter: str = KEY.EDGE_ATTR,
        data_key_weight_input: str = KEY.EDGE_EMBEDDING,
        data_key_edge_idx: str = KEY.EDGE_IDX,
        is_parallel: bool = False,
    ):
        super().__init__()
        self.denominator = nn.Parameter(
            torch.FloatTensor([denominator]), requires_grad=train_denominator
        )
        self.key_x = data_key_x
        self.key_filter = data_key_filter
        self.key_weight_input = data_key_weight_input
        self.key_edge_idx = data_key_edge_idx
        self.is_parallel = is_parallel

        self.irreps_x = irreps_x
        self.irreps_filter = irreps_filter
        self.irreps_out = irreps_out

        instructions = []
        irreps_mid = []
        for i, (mul_x, ir_x) in enumerate(irreps_x):
            for j, (_, ir_filter) in enumerate(irreps_filter):
                for ir_out in ir_x * ir_filter:
                    if ir_out in irreps_out:  # here we drop l > lmax
                        k = len(irreps_mid)
                        irreps_mid.append((mul_x, ir_out))
                        instructions.append((i, j, k, ir_out.dim**0.5))

        print(instructions)
        irreps_mid = Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort()  # type: ignore
        self.irreps_mid = irreps_mid
        instructions = [
            (i_in1, i_in2, p[i_out], pw)
            for i_in1, i_in2, i_out, pw in instructions
        ]
        self.instructions = sorted(instructions, key=lambda x: x[2])

        lmax = irreps_out.lmax
        coeff_dict = {}
        for irreps1 in irreps_x:
            l1 = irreps1.ir.l
            blocks = []
            l_list = []
            for l2 in range(irreps_filter.lmax + 1):
                w3j_list = []
                for l3 in range(abs(l1 - l2), l1 + l2 + 1):
                    if l3 > lmax:
                        continue
                    l_list.append((l1, l2, l3))
                    w3j_list.append(wigner_3j(l1, l2, l3))
                blocks.append(
                    torch.cat(w3j_list, axis=2)  # type: ignore
                    .transpose(0, 1)
                    .flatten(0, 1)
                )
            coeff_dict[l1] = (torch.block_diag(*blocks), l_list)
        self.w3j_dict = coeff_dict

        w_numel = sum([ir_x.mul for ir_x in irreps_x])
        indi = []
        mul_prev = 0
        for irreps in irreps_x:
            ir_elem = irreps.ir.dim
            mul = irreps.mul
            _indi = list(
                chain(*[[ii] * ir_elem for ii in range(mul_prev, mul_prev + mul)])
            )
            indi.append(_indi)
            mul_prev += mul
        indi = list(chain(*indi))
        self.w_indi = indi

        self.weight_nn_kwargs = dict(
            hs=weight_layer_input_to_hidden + [w_numel], act=weight_layer_act
        )

        self.weight_nn = FullyConnectedNet(**self.weight_nn_kwargs)

        self._comm_size = irreps_x.dim  # used in parallel

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        weight = self.weight_nn(data[self.key_weight_input])
        x = data[self.key_x]
        if self.is_parallel:
            x = torch.cat([x, data[KEY.NODE_FEATURE_GHOST]])
        y = data[self.key_filter]

        irreps_x, irreps_y, irreps_mid = (
            self.irreps_x,
            self.irreps_filter,
            self.irreps_mid,
        )

        # note that 1 -> src 0 -> dst
        edge_src = data[self.key_edge_idx][1]
        edge_dst = data[self.key_edge_idx][0]

        x_ex = x[edge_src] * weight[:, self.w_indi]  # [nedges, mul_ir]
        kron = torch.einsum('zi,zj->zij', y, x_ex)  # [nedges, mul_ir, ir]
        xx = (
            message_gather(x, edge_dst, kron) / self.denominator
        )  # [nnodes, mul_ir, ir]

        out_dct = {}
        for irreps_1, slice in zip(self.irreps_x, self.irreps_x.slices()):
            xx_ = (
                xx[:, :, slice]
                .reshape(-1, irreps_y.dim, irreps_1.mul, irreps_1.ir.dim)
                .transpose(1, 2)  # y_ir channel swap
                .flatten(2, 3)  # y_ir x_ir flatten
            )
            coeff, l_list = self.w3j_dict[irreps_1.ir.l]
            coeff = coeff.to(x.device)
            out = torch.einsum('zui,ik->zuk', xx_, coeff)  # nbatch, channel, ir_mul
            idx = 0
            for l1, l2, l3 in l_list:
                dim = 2 * l3 + 1
                out_dct[(l1, l2, l3)] = out[:, :, idx : idx + dim]
                idx += dim

        outputs = []
        for inst in self.instructions:
            l1 = irreps_x[inst[0]].ir.l  # type: ignore
            l2 = irreps_y[inst[1]].ir.l  # type: ignore
            l3 = irreps_mid[inst[2]].ir.l  # type: ignore
            outputs.append(out_dct[(l1, l2, l3)].flatten(1, 2) * inst[3])

        x = torch.cat(outputs, dim=1)

        if self.is_parallel:
            x = torch.tensor_split(x, data[KEY.NLOCAL])[0]
        data[self.key_x] = x
        return data
