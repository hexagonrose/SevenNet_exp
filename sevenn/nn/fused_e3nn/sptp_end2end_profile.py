import torch
import e3nn
import json
import cuequivariance as cue
from fused_e3nn import fused_uvu_TP
import os,sys
import cuequivariance as cue
import cuequivariance_torch as cuet

def mul_Irreps(mul, i_in):
    dd = []
    for ori_mul, ir in i_in:
        dd.append((ori_mul*mul, (ir.l, ir.p)))
    return e3nn.o3.Irreps(dd)
def compare(a, b):
    isclose = torch.isclose(a, b)
    diff_pos = torch.argwhere(isclose == False)
    anything_bad = False
    for pos in diff_pos:
        pos_t = [x for x in pos]
        if(abs(a[pos_t] - b[pos_t]) > 1e-6):
            anything_bad = True
            print(pos)
            print(a[pos_t] - b[pos_t] )
    if(not anything_bad):
        print("All Good")
            
IR_IN1_IDX = 0
IR_IN2_IDX = 1
IR_OUT_IDX = 2
INST_IDX = 3
WARPSIZE = 32

def load_nequip_config(h, l_max, layer_idx):
    filename = f"/home2/lsy/mdsim/nequip/benchmark_config/4_{h}_{l_max}_p_sc.txt"
    with open(filename, "r") as f:
        f_in = f.read().split("\n")

    per_layer_dict = dict()
    for l_idx, d in enumerate(f_in):
        if(d == "") : continue
        dd = json.loads(d)
        per_layer_dict[l_idx] = dd
    tp_list = per_layer_dict[layer_idx]["tp"]
    i_in1 = e3nn.o3.Irreps(tp_list[IR_IN1_IDX])
    i_in2 = e3nn.o3.Irreps(tp_list[IR_IN2_IDX])
    i_out = e3nn.o3.Irreps(tp_list[IR_OUT_IDX])
    inst_tuple = [tuple(x) for x in tp_list[INST_IDX]]

    return i_in1, i_in2, i_out, inst_tuple

def load_nequip_config_e3nn_cueq(h, l_max, layer_idx):
    filename = f"/home2/lsy/mdsim/nequip/benchmark_config/4_{h}_{l_max}_p_sc.txt"
    with open(filename, "r") as f:
        f_in = f.read().split("\n")

    per_layer_dict = dict()
    for l_idx, d in enumerate(f_in):
        if(d == "") : continue
        dd = json.loads(d)
        per_layer_dict[l_idx] = dd
    tp_list = per_layer_dict[layer_idx]["tp"]

    ei_in1 = e3nn.o3.Irreps(tp_list[IR_IN1_IDX])
    ei_in2 = e3nn.o3.Irreps(tp_list[IR_IN2_IDX])
    ei_out = e3nn.o3.Irreps(tp_list[IR_OUT_IDX])
    inst_tuple = [tuple(x) for x in tp_list[INST_IDX]]


    # changing mul for each ir.l
    new_in1_list = []
    new_out_list = []
    changed_idx = [[],[]]
    # mul_list = {}
    mul_list = {0:128, 1:64}

    for idx, (mul,ir) in enumerate(ei_in1):
        if (ir.l in mul_list):
            new_in1_list.append((mul_list[ir.l], ir))
            for inst in inst_tuple:
                if(idx == inst[0]):
                    changed_idx[0].append(inst[2])
                    changed_idx[1].append(mul_list[ir.l])
        else:
            new_in1_list.append((mul, ir))

    for idx, (mul,ir) in enumerate(ei_out):
        if (idx in changed_idx[0]):
            new_out_list.append((changed_idx[1][changed_idx[0].index(idx)], ir))
        else:
            new_out_list.append((mul, ir))

    ei_in1 = e3nn.o3.Irreps(new_in1_list)
    ei_out = e3nn.o3.Irreps(new_out_list)

    ci_in1 = cue.Irreps("O3", str(ei_in1))
    ci_in2 = cue.Irreps("O3", tp_list[IR_IN2_IDX])
    ci_out = cue.Irreps("O3", str(ei_out))


    return [ei_in1,ei_in2,ei_out,inst_tuple] , [ci_in1,ci_in2,ci_out,inst_tuple]

def main():
    h = int(sys.argv[1])
    l_max = int(sys.argv[2])
    layer_idx = int(sys.argv[3])
    tp_type = sys.argv[4]
    batch_size = int(sys.argv[5])
    use_compare = sys.argv[6] == "True"
    block_batch_cnt = int(sys.argv[7])

    e3nn_config, cueq_config = load_nequip_config_e3nn_cueq(h,l_max,layer_idx)
    i_in1, i_in2, i_out, inst_tuple = e3nn_config

    # # uvu
    tp = e3nn.o3.TensorProduct(i_in1,i_in2,i_out,inst_tuple,shared_weights=False, internal_weights=False) # path_normalization="none", normalization="none"
    tp = tp.to(device="cuda")

    cuet_tp = cuet.ChannelWiseTensorProduct(*cueq_config[:-1], shared_weights=False,internal_weights=False, device="cuda", layout="ir_mul")
    fused_tp = fused_uvu_TP(i_in1,i_in2,i_out,inst_tuple, per_block_batch=block_batch_cnt, device="cuda")


    torch.cuda.cudart().cudaProfilerStart()
    for i in range(10):
        in1 = torch.rand(batch_size, i_in1.dim, device="cuda", requires_grad=True)
        in2 = torch.rand(batch_size, i_in2.dim, device="cuda", requires_grad=True)
        weight = torch.rand(batch_size,tp.weight_numel, device="cuda", requires_grad=True)

        in1_c = torch.rand(batch_size, i_in1.dim, device="cuda", requires_grad=True)
        in2_c = torch.rand(batch_size, i_in2.dim, device="cuda", requires_grad=True)
        weight_c = torch.rand(batch_size,tp.weight_numel, device="cuda", requires_grad=True)

        cin1_cuda = torch.rand(batch_size, i_in1.dim, device="cuda", requires_grad=True)
        cin2_cuda = torch.rand(batch_size, i_in2.dim, device="cuda", requires_grad=True)
        weight_cueq = torch.rand(batch_size,tp.weight_numel, device="cuda", requires_grad=True)

        # fwd path
        torch.cuda.nvtx.range_push(f"e3nn_fwd")
        out_exp = tp(in1,in2,weight)
        torch.cuda.nvtx.range_pop()
        y = torch.nn.functional.gelu(out_exp).sum()

        # original e3nn
        torch.cuda.nvtx.range_push(f"e3nn_bwd")
        f_in1, f_in2, f_weight = torch.autograd.grad(y, [in1,in2,weight], create_graph=True)
        torch.cuda.nvtx.range_push(f"e3nn_bwd_dummy")
        print(f_in1[0])
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_pop()

        f_in1_gelu = torch.nn.functional.gelu(f_in1)
        f_in2_gelu = torch.nn.functional.gelu(f_in2)
        f_weight_gelu = torch.nn.functional.gelu(f_weight)
        fake_loss = f_in1_gelu.sum() + f_in2_gelu.sum() + f_weight_gelu.sum()

        torch.cuda.nvtx.range_push(f"e3nn_bwd_bwd+bwd")
        fake_loss.backward()
        torch.cuda.nvtx.range_push(f"e3nn_bwd_bwd+bwd_dummy")
        print(in1.grad[0])
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_pop()


        #### ours 

        torch.cuda.nvtx.range_push(f"sptp_fwd")
        out_ours = fused_tp(in1_c,in2_c,weight_c)
        torch.cuda.nvtx.range_pop()
        y_ours = torch.nn.functional.gelu(out_ours).sum()

        # original e3nn
        torch.cuda.nvtx.range_push(f"sptp_bwd")
        f_in1_c, f_in2_c, f_weight_c = torch.autograd.grad(y_ours, [in1_c,in2_c,weight_c], create_graph=True)
        torch.cuda.nvtx.range_push(f"sptp_bwd_dummy")
        print(f_in1_c[0])
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_pop()

        f_in1_gelu_c = torch.nn.functional.gelu(f_in1_c)
        f_in2_gelu_c = torch.nn.functional.gelu(f_in2_c)
        f_weight_gelu_c = torch.nn.functional.gelu(f_weight_c)
        fake_loss_c = f_in1_gelu_c.sum() + f_in2_gelu_c.sum() + f_weight_gelu_c.sum()
        
        torch.cuda.nvtx.range_push(f"sptp_bwd_bwd+bwd")
        fake_loss_c.backward()
        torch.cuda.nvtx.range_push(f"sptp_bwd_bwd+bwd_dummy")
        print(in1_c.grad[0])
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_pop()


       
        torch.cuda.nvtx.range_push(f"cueq_irmul_fwd")
        cuet_out = cuet_tp(cin1_cuda,cin2_cuda,weight_cueq)
        torch.cuda.nvtx.range_pop()
        y_cu = torch.nn.functional.gelu(cuet_out).sum()

        torch.cuda.nvtx.range_push(f"cueq_irmul_bwd")
        f_in1_cu, f_in2_cu, f_weight_cu = torch.autograd.grad(y_cu, [cin1_cuda,cin2_cuda,weight_cueq], create_graph=True)
        torch.cuda.nvtx.range_push(f"cueq_irmul_bwd_dummy")
        print(f_in1_cu[0])
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_pop()

    
        f_in1_gelu_cu = torch.nn.functional.gelu(f_in1_cu)
        f_in2_gelu_cu = torch.nn.functional.gelu(f_in2_cu)
        f_weight_gelu_cu = torch.nn.functional.gelu(f_weight_cu)
        fake_loss_cu = f_in1_gelu_cu.sum() + f_in2_gelu_cu.sum() + f_weight_gelu_cu.sum()

        torch.cuda.nvtx.range_push(f"cueq_irmul_bwd_bwd+bwd")
        fake_loss_cu.backward()
        torch.cuda.nvtx.range_push(f"cueq_irmul_bwd_bwd+bwd_dummy")
        print(cin1_cuda.grad[0])
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_pop()

    torch.cuda.cudart().cudaProfilerStop()

if __name__ == "__main__":
    main()
