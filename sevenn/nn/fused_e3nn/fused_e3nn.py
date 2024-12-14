from torch.utils.cpp_extension import load
import os
import torch
from typing import List
import e3nn
import itertools


os.environ['TORCH_CUDA_ARCH_LIST'] = '8.9'

prefix = '/home/parkyutack/SevenNet/sevenn/nn/fused_e3nn'
sptp_linear = load(
    name='sptp_linear',
    sources=[
        f'{prefix}/sptp_linear_bwd.cpp',
        f'{prefix}/bwd_sptp_linear_shared.cu',
        f'{prefix}/fwd_sptp_linear_v2_shared.cu',
        f'{prefix}/bwd_bwd_sptp_linear_v2_shared.cu',
    ],
    extra_cuda_cflags=['-lineinfo'],
    verbose=True,
)


@torch.library.custom_op(
    'sptp_linear::sptp_linear_fwd_v2_shared',
    mutates_args=(),
    device_types='cuda',
)
def _(
    in1: torch.Tensor,
    in2: torch.Tensor,
    weight: torch.Tensor,
    t_in1_idxing: torch.Tensor,
    t_in1_ival: torch.Tensor,
    t_in1_related_path_idx: torch.Tensor,
    t_path_array1: torch.Tensor,
    t_path_array2: torch.Tensor,
    t_per_upath_fiber_start: torch.Tensor,
    t_path_weight: torch.Tensor,
    t_per_path_weight_pos: torch.Tensor,
    t_per_upath_fiber_array: torch.Tensor,
    t_unique_cg_val: torch.Tensor,
    upath_cnt: int,
    per_block_batch: int,
    max_ir_dim: int,
    out_size: int,
) -> torch.Tensor:
    batch_size = in1.shape[0]
    out = torch.empty((batch_size, out_size), device=in1.device, dtype=in1.dtype)

    sptp_linear.sptp_linear_fwd_v2_shared(
        in1,
        in2,
        weight,
        out,
        t_in1_idxing,
        t_in1_ival,
        t_in1_related_path_idx,
        t_path_array1,
        t_path_array2,
        t_per_upath_fiber_start,
        t_path_weight,
        t_per_path_weight_pos,
        t_per_upath_fiber_array,
        t_unique_cg_val,
        upath_cnt,
        per_block_batch,
        max_ir_dim * 2 + 1,
    )
    return out


def fused_e3nn_setup_fwd_context(ctx, inputs, output):
    (
        in1,
        in2,
        weight,
        t_in1_idxing,
        t_in1_ival,
        t_in1_related_path_idx,
        t_path_array1,
        t_path_array2,
        t_per_upath_fiber_start,
        t_path_weight,
        t_per_path_weight_pos,
        t_per_upath_fiber_array,
        t_unique_cg_val,
        upath_cnt,
        per_block_batch,
        max_ir_dim,
        out_size,
    ) = inputs
    ctx.save_for_backward(
        in1,
        in2,
        weight,
        t_in1_idxing,
        t_in1_ival,
        t_in1_related_path_idx,
        t_path_array1,
        t_path_array2,
        t_per_upath_fiber_start,
        t_path_weight,
        t_per_path_weight_pos,
        t_per_upath_fiber_array,
        t_unique_cg_val,
    )
    ctx.upath_cnt = upath_cnt
    ctx.per_block_batch = per_block_batch
    ctx.max_ir_dim = max_ir_dim


def fused_e3nn_bwd(ctx, grad_output):
    (
        in1,
        in2,
        weight,
        t_in1_idxing,
        t_in1_ival,
        t_in1_related_path_idx,
        t_path_array1,
        t_path_array2,
        t_per_upath_fiber_start,
        t_path_weight,
        t_per_path_weight_pos,
        t_per_upath_fiber_array,
        t_unique_cg_val,
    ) = ctx.saved_tensors

    grad_list = torch.ops.sptp_linear.sptp_linear_bwd_v2_shared(
        in1,
        in2,
        weight,
        grad_output,
        t_in1_idxing,
        t_in1_ival,
        t_in1_related_path_idx,
        t_path_array1,
        t_path_array2,
        t_per_upath_fiber_start,
        t_path_weight,
        t_per_path_weight_pos,
        t_per_upath_fiber_array,
        t_unique_cg_val,
        ctx.upath_cnt,
        ctx.per_block_batch,
        ctx.max_ir_dim,
    )

    return (
        grad_list[0],  # in1_grad
        grad_list[1],  # in2_grad
        grad_list[2],  # weight_grad
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


@torch.library.custom_op(
    'sptp_linear::sptp_linear_bwd_v2_shared',
    mutates_args=(),
    device_types='cuda',
)
def _(
    in1: torch.Tensor,
    in2: torch.Tensor,
    weight: torch.Tensor,
    grad_output: torch.Tensor,
    t_in1_idxing: torch.Tensor,
    t_in1_ival: torch.Tensor,
    t_in1_related_path_idx: torch.Tensor,
    t_path_array1: torch.Tensor,
    t_path_array2: torch.Tensor,
    t_per_upath_fiber_start: torch.Tensor,
    t_path_weight: torch.Tensor,
    t_per_path_weight_pos: torch.Tensor,
    t_per_upath_fiber_array: torch.Tensor,
    t_unique_cg_val: torch.Tensor,
    upath_cnt: int,
    per_block_batch: int,
    max_ir_dim: int,
) -> List[torch.Tensor]:
    batch_size = in1.shape[0]
    in2_size = in2.shape[1]

    mem_debug = torch.empty((1, 1), device=in1.device)
    mem_dl_din1 = torch.empty_like(in1)
    mem_dl_din2 = torch.empty(
        (batch_size, in2_size * upath_cnt), device=in1.device, dtype=in1.dtype
    )
    mem_dl_dw = torch.empty_like(weight)

    sptp_linear.sptp_linear_bwd_v1_shared(
        in1,
        in2,
        weight,
        grad_output.contiguous(),
        mem_dl_din1,
        mem_dl_din2,
        mem_dl_dw,
        mem_debug,
        t_in1_idxing,
        t_in1_ival,
        t_in1_related_path_idx,
        t_path_array1,
        t_path_array2,
        t_per_upath_fiber_start,
        t_path_weight,
        t_per_path_weight_pos,
        t_per_upath_fiber_array,
        t_unique_cg_val,
        upath_cnt,
        per_block_batch,
        max_ir_dim * 2 + 1,
    )
    mem_dl_din2_summed = mem_dl_din2.reshape((batch_size, upath_cnt, in2_size)).sum(
        dim=1
    )

    return [mem_dl_din1, mem_dl_din2_summed, mem_dl_dw]


@torch.library.custom_op(
    'sptp_linear::sptp_linear_bwd_bwd_v2_shared',
    mutates_args=(),
    device_types='cuda',
)
def _(
    dF_in1: torch.Tensor,
    dF_in2: torch.Tensor,
    dF_dw: torch.Tensor,
    dE_dout: torch.Tensor,
    in1: torch.Tensor,
    in2: torch.Tensor,
    weight: torch.Tensor,
    t_in1_idxing: torch.Tensor,
    t_in1_ival: torch.Tensor,
    t_in1_related_path_idx: torch.Tensor,
    t_path_array1: torch.Tensor,
    t_path_array2: torch.Tensor,
    t_per_upath_fiber_start: torch.Tensor,
    t_path_weight: torch.Tensor,
    t_per_path_weight_pos: torch.Tensor,
    t_per_upath_fiber_array: torch.Tensor,
    t_unique_cg_val: torch.Tensor,
    upath_cnt: int,
    per_block_batch: int,
    max_ir_dim: int,
) -> List[torch.Tensor]:
    batch_size = in2.shape[0]
    in2_size = in2.shape[1]

    dF_dout = torch.empty_like(dE_dout)
    dL_din1 = torch.empty_like(in1)
    dL_din2_duplicate = torch.empty(
        (batch_size, in2_size * upath_cnt), device=in2.device, dtype=in2.dtype
    )
    dL_dw = torch.empty_like(weight)
    mem_debug = torch.empty((1, 1), device=in1.device)

    sptp_linear.sptp_linear_bwd_bwd_v2_shared(
        dF_in1,
        dF_in2,
        dF_dw,
        dE_dout,
        in1,
        in2,
        weight,
        dF_dout,
        dL_dw,
        dL_din1,
        dL_din2_duplicate,
        mem_debug,
        t_in1_idxing,
        t_in1_ival,
        t_in1_related_path_idx,
        t_path_array1,
        t_path_array2,
        t_per_upath_fiber_start,
        t_path_weight,
        t_per_path_weight_pos,
        t_per_upath_fiber_array,
        t_unique_cg_val,
        upath_cnt,
        per_block_batch,
        max_ir_dim * 2 + 1,
    )

    dL_din2 = dL_din2_duplicate.reshape((batch_size, upath_cnt, in2_size)).sum(dim=1)

    return [dL_din1, dL_din2, dL_dw, dF_dout]


def fused_e3nn_setup_bwd_context(ctx, inputs, output):
    (
        in1,
        in2,
        weight,
        dE_dout,
        t_in1_idxing,
        t_in1_ival,
        t_in1_related_path_idx,
        t_path_array1,
        t_path_array2,
        t_per_upath_fiber_start,
        t_path_weight,
        t_per_path_weight_pos,
        t_per_upath_fiber_array,
        t_unique_cg_val,
        upath_cnt,
        per_block_batch,
        max_ir_dim,
    ) = inputs

    ctx.save_for_backward(
        dE_dout,
        in1,
        in2,
        weight,
        t_in1_idxing,
        t_in1_ival,
        t_in1_related_path_idx,
        t_path_array1,
        t_path_array2,
        t_per_upath_fiber_start,
        t_path_weight,
        t_per_path_weight_pos,
        t_per_upath_fiber_array,
        t_unique_cg_val,
    )
    ctx.upath_cnt = upath_cnt
    ctx.per_block_batch = per_block_batch
    ctx.max_ir_dim = max_ir_dim


def fused_e3nn_bwd_bwd(ctx, grad_output):
    (
        dE_dout,
        in1,
        in2,
        weight,
        t_in1_idxing,
        t_in1_ival,
        t_in1_related_path_idx,
        t_path_array1,
        t_path_array2,
        t_per_upath_fiber_start,
        t_path_weight,
        t_per_path_weight_pos,
        t_per_upath_fiber_array,
        t_unique_cg_val,
    ) = ctx.saved_tensors

    dF_in1 = grad_output[0]
    dF_in2 = grad_output[1]
    dF_w = grad_output[2]

    grad_list = torch.ops.sptp_linear.sptp_linear_bwd_bwd_v2_shared(
        dF_in1,
        dF_in2,
        dF_w,
        dE_dout.detach(),
        in1,
        in2,
        weight,
        t_in1_idxing,
        t_in1_ival,
        t_in1_related_path_idx,
        t_path_array1,
        t_path_array2,
        t_per_upath_fiber_start,
        t_path_weight,
        t_per_path_weight_pos,
        t_per_upath_fiber_array,
        t_unique_cg_val,
        ctx.upath_cnt,
        ctx.per_block_batch,
        ctx.max_ir_dim,
    )

    return (
        grad_list[0],
        grad_list[1],
        grad_list[2],  # weight_grad
        grad_list[3],  # mem_dL_dO_grad
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


torch.library.register_autograd(
    'sptp_linear::sptp_linear_fwd_v2_shared',
    fused_e3nn_bwd,
    setup_context=fused_e3nn_setup_fwd_context,
)

torch.library.register_autograd(
    'sptp_linear::sptp_linear_bwd_v2_shared',
    fused_e3nn_bwd_bwd,
    setup_context=fused_e3nn_setup_bwd_context,
)


class fused_uvu_TP(torch.nn.Module):
    def __init__(
        self,
        i_in1,
        i_in2,
        i_out,
        inst_tuple,
        warpsize=32,
        per_block_batch=4,
        device='cuda',
        shared_weights=False,
        internal_weights=False,
    ):
        super().__init__()
        self.i_in1 = i_in1
        self.i_in2 = i_in2
        self.i_out = i_out
        self.WARPSIZE = warpsize
        self.device = device

        self.per_block_batch = per_block_batch
        self.l_max = max(
            [
                max([x[1].l for x in i_in1]),
                max([x[1].l for x in i_in2]),
                max([x[1].l for x in i_out]),
            ]
        )

        _irreps_out = [e3nn.o3.Irrep(o.ir.l, o.ir.p) for o in i_out]
        uvuv_tp = e3nn.o3.FullTensorProduct(i_in1, i_in2, filter_ir_out=_irreps_out)
        uvuv_i_out = uvuv_tp.irreps_out
        tp = e3nn.o3.TensorProduct(
            i_in1,
            i_in2,
            i_out,
            inst_tuple,
            shared_weights=shared_weights,
            internal_weights=internal_weights,
        )

        unique_cg, unique_cg_mat = self.extract_cg_info(
            uvuv_i_out, uvuv_tp.instructions
        )

        self.unique_cg_val = list(set([x.item() for x in unique_cg]))
        tp_inst_outorder = sorted(tp.instructions, key=lambda x: x.i_out)

        per_path_fiber_start, per_path_fiber_array, per_in1_ir_pathinfo = (
            self.cgmat2fiber(tp_inst_outorder, unique_cg_mat)
        )

        self.metadata_list = self.metadata_gen(
            tp.instructions,
            per_path_fiber_start,
            per_path_fiber_array,
            per_in1_ir_pathinfo,
        )

    def forward(self, in1, in2, weight):
        out = torch.ops.sptp_linear.sptp_linear_fwd_v2_shared(
            in1,
            in2,
            weight,
            *self.metadata_list,
            self.per_block_batch,
            self.l_max,
            self.i_out.dim,
        )
        return out

    def extract_cg_info(self, uvuv_i_out, instructions):
        unique_cg = []
        unique_cg_mat = {}
        for inst in instructions:
            i = inst.i_in1
            j = inst.i_in2
            k = inst.i_out

            mul_in1, ir_in1 = self.i_in1[i]
            mul_in2, ir_in2 = self.i_in2[j]
            mul_out, ir_out = uvuv_i_out[k]

            cg = e3nn.o3.wigner_3j(ir_in1.l, ir_in2.l, ir_out.l)
            unique_cg += list(cg.unique())

            partial_mat_cg = torch.zeros(
                self.i_in1[i].dim, self.i_in2[j].dim, uvuv_i_out[k].dim
            )
            # print(cg)
            unique_cg_mat[f'{ir_in1.l}_{ir_in2.l}_{ir_out.l}'] = cg

            ## uvuv
            for u, v in itertools.product(range(mul_in1), range(mul_in2)):
                partial_mat_cg[
                    u * ir_in1.dim : (u + 1) * ir_in1.dim,
                    v * ir_in2.dim : (v + 1) * ir_in2.dim,
                    (u * mul_in2 + v) * ir_out.dim : (u * mul_in2 + v + 1)
                    * ir_out.dim,
                ] = cg

        return unique_cg, unique_cg_mat

    def cgmat2fiber(self, tp_inst_outorder, unique_cg_mat):
        per_path_fiber_start = [0]
        per_path_fiber_array = []
        per_in1_ir_pathinfo = {}

        for inst in tp_inst_outorder:
            path_cg = unique_cg_mat[
                f'{self.i_in1[inst.i_in1][1].l}_{self.i_in2[inst.i_in2][1].l}_{self.i_out[inst.i_out][1].l}'
            ]
            for i, j, k in path_cg.nonzero():
                cg_idx = self.unique_cg_val.index(path_cg[i, j, k])
                per_path_fiber_array.append([i.item(), j.item(), k.item(), cg_idx])
            per_path_fiber_start.append(len(per_path_fiber_array))

            if inst.i_in1 not in per_in1_ir_pathinfo:
                per_in1_ir_pathinfo[inst.i_in1] = []
            per_in1_ir_pathinfo[inst.i_in1].append(
                [inst.i_out, inst.i_in2, inst.path_weight]
            )

        return per_path_fiber_start, per_path_fiber_array, per_in1_ir_pathinfo

    def metadata_gen(
        self,
        instructions,
        per_path_fiber_start,
        per_path_fiber_array,
        per_in1_ir_pathinfo,
    ):
        weight_uv_pair = []
        weight_uv_pair_sorted_chunk = []
        out_order = []
        current = 0

        for inst in instructions:
            weight_uv_pair.append(
                (self.i_in1[inst.i_in1][0], self.i_in2[inst.i_in2][0])
            )
            out_order.append(inst.i_out)
        for u, v in weight_uv_pair:
            weight_uv_pair_sorted_chunk.append(slice(current, current + u * v))
            current += u * v
        out2weight_order = torch.tensor(out_order).argsort()

        in1_idxing = [0]
        in1_ival = []
        in1_related_path_idx = [0]

        path_array1 = []
        path_array2 = []
        path_weight = []
        per_path_weight_pos = []

        per_upath_fiber_start = []
        per_upath_fiber_array = []

        in1_slices = self.i_in1.slices()
        in2_slices = self.i_in2.slices()
        out_slices = self.i_out.slices()

        for in1_ir_idx, (mul, ir) in enumerate(self.i_in1):
            assert mul % self.WARPSIZE == 0
            in1_idx_start = in1_slices[in1_ir_idx].start
            i_val = ir.dim

            if mul >= self.WARPSIZE:
                for i in range(mul // self.WARPSIZE):
                    in1_idxing.append(
                        in1_idx_start + self.WARPSIZE * i_val * (i + 1)
                    )
                    in1_ival.append(i_val)
                    in1_related_path_idx.append(
                        in1_related_path_idx[-1]
                        + len(per_in1_ir_pathinfo[in1_ir_idx])
                    )

                    dummy_list = []
                    dummy_list2 = []
                    # Bug? TODO:
                    for out_ir_idx, in2_ir_idx, pw in per_in1_ir_pathinfo[
                        in1_ir_idx
                    ]:
                        # should be in order
                        fiber_start = per_path_fiber_start[out_ir_idx]
                        fiber_end = per_path_fiber_start[out_ir_idx + 1]

                        upath_fiber_start = len(per_upath_fiber_array)
                        upath_fiber_end = upath_fiber_start + fiber_end - fiber_start

                        per_upath_fiber_start.append(
                            [upath_fiber_start, upath_fiber_end]
                        )
                        # print(fiber_array_orignal[1:4])
                        # print(fiber_start,fiber_end)
                        # print(fiber_array_orignal[fiber_start:fiber_end])

                        per_upath_fiber_array += per_path_fiber_array[
                            fiber_start:fiber_end
                        ]

                        dummy_list.append(
                            [
                                out_slices[out_ir_idx].start
                                + self.WARPSIZE * self.i_out[out_ir_idx].ir.dim * i,
                                out_slices[out_ir_idx].start
                                + self.WARPSIZE
                                * self.i_out[out_ir_idx].ir.dim
                                * (i + 1),
                            ]
                        )
                        dummy_list2.append(
                            [
                                self.i_out[out_ir_idx].ir.dim,
                                in2_slices[in2_ir_idx].start,
                                self.i_in2[in2_ir_idx].ir.dim,
                                in2_slices[in2_ir_idx].stop,
                            ]
                        )
                        path_weight.append(pw)

                        # TODO:??
                        per_path_weight_pos.append(
                            weight_uv_pair_sorted_chunk[
                                out2weight_order[out_ir_idx]
                            ].start
                            + self.WARPSIZE * i
                        )

                    path_array1.append(dummy_list)
                    path_array2.append(dummy_list2)

        t_in1_idxing = torch.tensor(
            in1_idxing, dtype=torch.int32, device=self.device
        )
        t_in1_ival = torch.tensor(in1_ival, dtype=torch.int32, device=self.device)
        t_in1_related_path_idx = torch.tensor(
            in1_related_path_idx, dtype=torch.int32, device=self.device
        )

        t_path_array1 = torch.tensor(
            list(itertools.chain.from_iterable(path_array1)),
            dtype=torch.uint16,
            device=self.device,
        )
        t_path_array2 = torch.tensor(
            list(itertools.chain.from_iterable(path_array2)),
            dtype=torch.uint8,
            device=self.device,
        )
        t_path_weight = torch.tensor(
            path_weight, dtype=torch.float32, device=self.device
        )
        t_per_path_weight_pos = torch.tensor(
            per_path_weight_pos, dtype=torch.int32, device=self.device
        )

        t_per_upath_fiber_start = torch.tensor(
            per_upath_fiber_start, dtype=torch.uint16, device=self.device
        )
        t_per_upath_fiber_array = torch.tensor(
            per_upath_fiber_array, dtype=torch.uint8, device=self.device
        )
        t_unique_cg_val = torch.tensor(
            self.unique_cg_val, dtype=torch.float32, device=self.device
        )
        upath_cnt = len(in1_idxing) - 1

        return [
            t_in1_idxing,
            t_in1_ival,
            t_in1_related_path_idx,
            t_path_array1,
            t_path_array2,
            t_per_upath_fiber_start,
            t_path_weight,
            t_per_path_weight_pos,
            t_per_upath_fiber_array,
            t_unique_cg_val,
            upath_cnt,
        ]


__all__ = ['fused_uvu_TP']
