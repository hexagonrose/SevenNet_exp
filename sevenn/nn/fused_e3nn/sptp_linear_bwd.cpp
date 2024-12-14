#include <torch/extension.h>
#include "sptp.hpp"
#include <iostream>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)



// void sptp_linear_bwd_v1(    
//     torch::Tensor in1, 
//     torch::Tensor in2,
//     torch::Tensor weight,
//     torch::Tensor mem_dL_dO,
//     torch::Tensor mem_dL_din1,
//     torch::Tensor mem_dL_din2,
//     torch::Tensor mem_dL_dW,
//     torch::Tensor mem_debug,

//     torch::Tensor t_in1_idxing,
//     torch::Tensor t_in1_ival,
//     torch::Tensor t_in1_related_path_idx,

//     torch::Tensor t_path_array1,
//     torch::Tensor t_path_array2,
//     torch::Tensor t_per_path_fiber_start,
//     torch::Tensor t_path_weight,
//     torch::Tensor t_per_path_weight_pos,

//     torch::Tensor t_fiber_array,
//     torch::Tensor t_unique_cg_val,

//     size_t path_cnt,
//     size_t per_block_batch
// ){

//     CHECK_INPUT(in1);
//     CHECK_INPUT(in2);
//     CHECK_INPUT(weight);
//     CHECK_INPUT(mem_dL_dO);
//     CHECK_INPUT(mem_dL_din1);
//     CHECK_INPUT(mem_dL_din2);
//     CHECK_INPUT(mem_dL_dW);

//     CHECK_INPUT(t_in1_idxing);
//     CHECK_INPUT(t_in1_ival);
//     CHECK_INPUT(t_in1_related_path_idx);

//     CHECK_INPUT(t_path_array1);
//     CHECK_INPUT(t_path_array2);
//     CHECK_INPUT(t_per_path_fiber_start);
//     CHECK_INPUT(t_path_weight);
//     CHECK_INPUT(t_per_path_weight_pos);

//     CHECK_INPUT(t_fiber_array);
//     CHECK_INPUT(t_unique_cg_val);
    
//     bwd_sptp_linear_cuda_v1(
//          in1, 
//          in2,
//          weight,
//          mem_dL_dO,
//          mem_dL_din1,
//          mem_dL_din2,
//          mem_dL_dW,
//          mem_debug,

//          t_in1_idxing,
//          t_in1_ival,
//          t_in1_related_path_idx,

//          t_path_array1,
//          t_path_array2,
//          t_per_path_fiber_start,
//          t_path_weight,
//          t_per_path_weight_pos,

//          t_fiber_array,
//          t_unique_cg_val,

//         path_cnt,
//         per_block_batch
//     );
// }

void sptp_linear_bwd_v1_shared(    
    torch::Tensor in1, 
    torch::Tensor in2,
    torch::Tensor weight,
    torch::Tensor mem_dL_dO,
    torch::Tensor mem_dL_din1,
    torch::Tensor mem_dL_din2,
    torch::Tensor mem_dL_dW,
    torch::Tensor mem_debug,

    torch::Tensor t_in1_idxing,
    torch::Tensor t_in1_ival,
    torch::Tensor t_in1_related_path_idx,

    torch::Tensor t_path_array1,
    torch::Tensor t_path_array2,
    torch::Tensor t_per_path_fiber_start,
    torch::Tensor t_path_weight,
    torch::Tensor t_per_path_weight_pos,

    torch::Tensor t_fiber_array,
    torch::Tensor t_unique_cg_val,

    size_t path_cnt,
    size_t per_block_batch,
    size_t max_ir_dim
){

    CHECK_INPUT(in1);
    CHECK_INPUT(in2);
    CHECK_INPUT(weight);
    CHECK_INPUT(mem_dL_dO);
    CHECK_INPUT(mem_dL_din1);
    CHECK_INPUT(mem_dL_din2);
    CHECK_INPUT(mem_dL_dW);

    CHECK_INPUT(t_in1_idxing);
    CHECK_INPUT(t_in1_ival);
    CHECK_INPUT(t_in1_related_path_idx);

    CHECK_INPUT(t_path_array1);
    CHECK_INPUT(t_path_array2);
    CHECK_INPUT(t_per_path_fiber_start);
    CHECK_INPUT(t_path_weight);
    CHECK_INPUT(t_per_path_weight_pos);

    CHECK_INPUT(t_fiber_array);
    CHECK_INPUT(t_unique_cg_val);
    
    bwd_sptp_linear_cuda_v1_shared(
         in1, 
         in2,
         weight,
         mem_dL_dO,
         mem_dL_din1,
         mem_dL_din2,
         mem_dL_dW,
         mem_debug,

         t_in1_idxing,
         t_in1_ival,
         t_in1_related_path_idx,

         t_path_array1,
         t_path_array2,
         t_per_path_fiber_start,
         t_path_weight,
         t_per_path_weight_pos,

         t_fiber_array,
         t_unique_cg_val,

        path_cnt,
        per_block_batch,
        max_ir_dim
    );
}

// void sptp_linear_fwd_v2(    
//     torch::Tensor in1, 
//     torch::Tensor in2,
//     torch::Tensor weight,
//     torch::Tensor out,

//     torch::Tensor t_in1_idxing,
//     torch::Tensor t_in1_ival,
//     torch::Tensor t_in1_related_path_idx,

//     torch::Tensor t_path_array1,
//     torch::Tensor t_path_array2,
//     torch::Tensor t_per_path_fiber_start,
//     torch::Tensor t_path_weight,
//     torch::Tensor t_per_path_weight_pos,

//     torch::Tensor t_fiber_array,
//     torch::Tensor t_unique_cg_val,

//     size_t path_cnt,
//     size_t per_block_batch
// ){

//     CHECK_INPUT(in1);
//     CHECK_INPUT(in2);
//     CHECK_INPUT(weight);
//     CHECK_INPUT(out);

//     CHECK_INPUT(t_in1_idxing);
//     CHECK_INPUT(t_in1_ival);
//     CHECK_INPUT(t_in1_related_path_idx);

//     CHECK_INPUT(t_path_array1);
//     CHECK_INPUT(t_path_array2);
//     CHECK_INPUT(t_per_path_fiber_start);
//     CHECK_INPUT(t_path_weight);
//     CHECK_INPUT(t_per_path_weight_pos);

//     CHECK_INPUT(t_fiber_array);
//     CHECK_INPUT(t_unique_cg_val);
    
//     fwd_sptp_linear_cuda_v2(
//          in1, 
//          in2,
//          weight,
//          out,
//          t_in1_idxing,
//          t_in1_ival,
//          t_in1_related_path_idx,

//          t_path_array1,
//          t_path_array2,
//          t_per_path_fiber_start,
//          t_path_weight,
//          t_per_path_weight_pos,

//          t_fiber_array,
//          t_unique_cg_val,

//         path_cnt,
//         per_block_batch
//     );
// }

void sptp_linear_fwd_v2_shared(    
    torch::Tensor in1, 
    torch::Tensor in2,
    torch::Tensor weight,
    torch::Tensor out,

    torch::Tensor t_in1_idxing,
    torch::Tensor t_in1_ival,
    torch::Tensor t_in1_related_path_idx,

    torch::Tensor t_path_array1,
    torch::Tensor t_path_array2,
    torch::Tensor t_per_path_fiber_start,
    torch::Tensor t_path_weight,
    torch::Tensor t_per_path_weight_pos,

    torch::Tensor t_fiber_array,
    torch::Tensor t_unique_cg_val,

    size_t path_cnt,
    size_t per_block_batch,
    size_t max_ir_dim
){

    CHECK_INPUT(in1);
    CHECK_INPUT(in2);
    CHECK_INPUT(weight);
    CHECK_INPUT(out);

    CHECK_INPUT(t_in1_idxing);
    CHECK_INPUT(t_in1_ival);
    CHECK_INPUT(t_in1_related_path_idx);

    CHECK_INPUT(t_path_array1);
    CHECK_INPUT(t_path_array2);
    CHECK_INPUT(t_per_path_fiber_start);
    CHECK_INPUT(t_path_weight);
    CHECK_INPUT(t_per_path_weight_pos);

    CHECK_INPUT(t_fiber_array);
    CHECK_INPUT(t_unique_cg_val);
    
    fwd_sptp_linear_cuda_v2_shared(
         in1, 
         in2,
         weight,
         out,

         t_in1_idxing,
         t_in1_ival,
         t_in1_related_path_idx,

         t_path_array1,
         t_path_array2,
         t_per_path_fiber_start,
         t_path_weight,
         t_per_path_weight_pos,

         t_fiber_array,
         t_unique_cg_val,

        path_cnt,
        per_block_batch,
        max_ir_dim
    );
}

void sptp_linear_bwd_bwd_v2_shared(    
    torch::Tensor mem_dF_din1, 
    torch::Tensor mem_dF_din2,
    torch::Tensor mem_dF_dW,
    torch::Tensor mem_dE_dO,

    torch::Tensor in1,
    torch::Tensor in2,
    torch::Tensor weight,
    
    torch::Tensor mem_dF_dO,
    torch::Tensor mem_dL_dW,
    torch::Tensor mem_dL_din1,
    torch::Tensor mem_dL_din2,
    torch::Tensor mem_debug,

    torch::Tensor t_in1_idxing,
    torch::Tensor t_in1_ival,
    torch::Tensor t_in1_related_path_idx,

    torch::Tensor t_path_array1,
    torch::Tensor t_path_array2,
    torch::Tensor t_per_path_fiber_start,
    torch::Tensor t_path_weight,
    torch::Tensor t_per_path_weight_pos,

    torch::Tensor t_fiber_array,
    torch::Tensor t_unique_cg_val,

    size_t path_cnt,
    size_t per_block_batch,
    size_t max_ir_dim
){
    CHECK_INPUT(mem_dF_din1);
    CHECK_INPUT(mem_dF_din2);
    CHECK_INPUT(mem_dF_dW);
    CHECK_INPUT(mem_dE_dO);

    CHECK_INPUT(in1);
    CHECK_INPUT(in2);
    CHECK_INPUT(weight);

    CHECK_INPUT(mem_dF_dO);
    CHECK_INPUT(mem_dL_dW);
    CHECK_INPUT(mem_dL_din1);
    CHECK_INPUT(mem_dL_din2);
    CHECK_INPUT(mem_debug);

    CHECK_INPUT(t_in1_idxing);
    CHECK_INPUT(t_in1_ival);
    CHECK_INPUT(t_in1_related_path_idx);

    CHECK_INPUT(t_path_array1);
    CHECK_INPUT(t_path_array2);
    CHECK_INPUT(t_per_path_fiber_start);
    CHECK_INPUT(t_path_weight);
    CHECK_INPUT(t_per_path_weight_pos);

    CHECK_INPUT(t_fiber_array);
    CHECK_INPUT(t_unique_cg_val);
    
    bwd_bwd_sptp_linear_cuda_v2_shared(
        mem_dF_din1, 
         mem_dF_din2,
         mem_dF_dW,
         mem_dE_dO,

         in1, 
         in2,
         weight,

         mem_dF_dO,
         mem_dL_dW,
         mem_dL_din1,
         mem_dL_din2,
         mem_debug,

         t_in1_idxing,
         t_in1_ival,
         t_in1_related_path_idx,

         t_path_array1,
         t_path_array2,
         t_per_path_fiber_start,
         t_path_weight,
         t_per_path_weight_pos,

         t_fiber_array,
         t_unique_cg_val,

        path_cnt,
        per_block_batch,
        max_ir_dim
    );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // m.def("sptp_linear_bwd_v1", &sptp_linear_bwd_v1, "sptp_linear_bwd_v1");
    // m.def("sptp_linear_fwd_v2", &sptp_linear_fwd_v2, "sptp_linear_fwd_v2");
    m.def("sptp_linear_bwd_v1_shared", &sptp_linear_bwd_v1_shared, "sptp_linear_bwd_v1_shared");
    m.def("sptp_linear_fwd_v2_shared", &sptp_linear_fwd_v2_shared, "sptp_linear_fwd_v2_shared");
    m.def("sptp_linear_bwd_bwd_v2_shared", &sptp_linear_bwd_bwd_v2_shared, "sptp_linear_bwd_bwd_v2_shared");
}
