#include "sptp.hpp"
#include <cmath>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, char const* const func, char const* const file,
           int const line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(char const* const file, int const line)
{
    cudaError_t const err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define MAX_IR 11 // up to L_max =5
#define MAX_IN2 36 // up to L_max =5
#define WARPSIZE 32 
#define MAX_IN1_IR_CNT 32 // not just len(i_in1) but if u is larger than 32 need multiple IR_CNT 
#define MAX_NUM_PATH 512 // also need to account for u > 32
#define MAX_U_FIBER_CNT 5265 // up to L_max =5
#define MAX_U_CG_VAL_CNT 344 // up to L_max =5

// struct __align__(8) path_struct {
//     const u_short k_start;
//     const u_short k_end;  
//     const u_char k_val;
//     const u_char j_start;
//     const u_char j_val;
//     const u_char j_end;
// };

// struct __align__(4) fiber_struct {
//     const u_char i_idx;
//     const u_char j_idx;
//     const u_char k_idx;
//     const u_char cg_idx;
// };

// struct __align__(4) fiber_idx_struct {
//     const u_short start_idx;
//     const u_short end_idx;  
// };

__constant__ int in1_idxing[MAX_IN1_IR_CNT];
__constant__ int in1_ival[MAX_IN1_IR_CNT];
__constant__ int in1_related_path_idx[MAX_IN1_IR_CNT];

__constant__ ushort2 path_array1[MAX_NUM_PATH];
__constant__ uchar4 path_array2[MAX_NUM_PATH];
__constant__ ushort2 per_path_fiber_start[MAX_NUM_PATH];
__constant__ float path_weight[MAX_NUM_PATH];
__constant__ int per_path_weight_pos[MAX_NUM_PATH];

__constant__ uchar4 fiber_array[MAX_U_FIBER_CNT];
__constant__ float unique_cg_val[MAX_U_CG_VAL_CNT];

template <typename scalar_t>
__global__ void bwd_sptp_lienar_kernel_v1(
    const float* __restrict__ in1,
    const float* __restrict__ in2,
    const float* __restrict__ weight,
    const float* __restrict__ mem_dL_dO,
    float* __restrict__ mem_dL_din1,
    float* __restrict__ mem_dL_din2,
    float* __restrict__ mem_dL_dW,
    float* __restrict__ mem_debug,

    const size_t batch_size,
    const size_t out_size,
    const size_t weight_size,
    const size_t in1_size,
    const size_t in2_size,
    const size_t path_cnt)
    {
    extern __shared__ scalar_t shmem[];
    // Input dL_dO => batch, ir, mul order
    // 2D grid, 2D block
    // grid (path, batch), block (mul(same path), batch)
    // intra-warp (u parallel) x , inter-warp (batch) y 
    const int target_in1 = blockIdx.x;
    const int global_t_batch_idx = blockIdx.y * blockDim.y + threadIdx.y;
    const int shmem_warp_pos_start = threadIdx.y*blockDim.x;

    if(global_t_batch_idx >= batch_size) return;

    // const int t_g_edge_dest = ;
    // const int t_g_edge_src = ;

    // check given path (path per thread_block)

    // start_end of out for a block
    // divide by u 

    // load all in2 to shmem
    // load all nnz fiber to shmem
    // load cg value (to register?)
    // load all w 
    // sync
    
    // no init needed just copy
    float reg_dldo[MAX_IR];
    float reg_in1[MAX_IR];
    float reg_in2[MAX_IR];
    
    // float reg_dL_dw[MAX_IR];
    // accumulated need init
    float reg_uvuv[MAX_IR];
    float reg_dL_din1[MAX_IR];
    float reg_dL_din2[MAX_IN2];

    scalar_t* shmem_in2 = shmem + threadIdx.y*(blockDim.x*MAX_IR+in2_size);
    scalar_t* shmem_dldo = shmem_in2 + in2_size;
    // dL_dO size : WARPSIZE * MAX_IR (all warps) * concurrent_batch (warp cnt)
    // dL_dO size : out_size 
    // (which i_in1 path, batch) (mul, batch)
    // need a lot of register.. (unless i make macro for all cases)

    // what defines the target_in1 ?? that is the question need z axis?

    // load part of in1 from main mem
    // in1 (z, mul, ir)
    const int in1_start = in1_idxing[target_in1];
    const int in1_end = in1_idxing[target_in1+1];
    const int i_val = in1_ival[target_in1];
    const int path_idx_start = in1_related_path_idx[target_in1];
    const int path_idx_end = in1_related_path_idx[target_in1+1];


    for(int shmem_idx = threadIdx.x, in1_idx = global_t_batch_idx*in1_size + in1_start+threadIdx.x; in1_idx < global_t_batch_idx*in1_size + in1_end; shmem_idx+=WARPSIZE, in1_idx+=WARPSIZE) {
        shmem_dldo[shmem_idx] = in1[in1_idx];
    }
    __syncwarp();
    for(int i =0, shmem_idx = threadIdx.x*i_val; i<i_val; i++, shmem_idx++){
        reg_in1[i] = shmem_dldo[shmem_idx];
    }

    // load all in2 from main mem
    for(int shmem_idx = threadIdx.x, in2_idx = global_t_batch_idx*in2_size + threadIdx.x; shmem_idx < in2_size; in2_idx+=WARPSIZE, shmem_idx+=WARPSIZE) {
        shmem_in2[shmem_idx] = in2[in2_idx];
    }
    __syncwarp();

    for(int i=0; i<MAX_IR;i++){
        reg_dL_din1[i] = 0.0;
    }
    for(int i=0; i<MAX_IN2;i++){
        reg_dL_din2[i] = 0.0;
    }

    // for path_chunk
    // path idx == k idx
    // path index == 
    const int g_t_dldo_start = global_t_batch_idx*out_size;
    const int g_t_w_start = global_t_batch_idx*weight_size;

    for(int path_idx=path_idx_start; path_idx < path_idx_end; path_idx++){
        const ushort2 path_info1 = path_array1[path_idx]; // k_start, k_end
        const uchar4 path_info2 = path_array2[path_idx]; // k_val, j_start, j_val, j_end

        for(int i=0; i<MAX_IR;i++){
            reg_uvuv[i] = 0.0;
        }
        
        // stall due to global memory access (better if it is load to shared memory and accessed)
        // possible optimization point with gather scatter
        for(int shmem_idx = threadIdx.x, dldo_idx = g_t_dldo_start + path_info1.x + threadIdx.x; dldo_idx < g_t_dldo_start+path_info1.y; dldo_idx+=WARPSIZE, shmem_idx+=WARPSIZE){
            shmem_dldo[shmem_idx] = mem_dL_dO[dldo_idx];
        }
        __syncwarp();
      
        // load k_val amount from shmem
        for(int i =0, shmem_idx = threadIdx.x*path_info2.x; i<path_info2.x; i++, shmem_idx++){
            reg_dldo[i] = shmem_dldo[shmem_idx];
        }
       
        // load in2 from shmem to reg
        for(int i =0, shmem_idx = path_info2.y; i<path_info2.z; i++, shmem_idx++){
            reg_in2[i] = shmem_in2[shmem_idx];
        }

        // odd number of k_val (2n+1) no bank conflict
        
        // Loading Weight from global memory is a major memory bottleneck
        const int weight_pos = g_t_w_start + per_path_weight_pos[path_idx]+threadIdx.x;
        float reg_w_path_norm = weight[weight_pos] * path_weight[path_idx];
        
        const ushort2 fiber_idx_info = per_path_fiber_start[path_idx];
        // for nnz in the fiber
        // uchar4 fiber;
        for(ushort fiber_idx = fiber_idx_info.x; fiber_idx < fiber_idx_info.y; fiber_idx++){
            // mult k with all w => dL_duvuv
            uchar4 fiber = fiber_array[fiber_idx]; // i, j, k, cg idx
            float dL_dOuter = reg_dldo[fiber.z] * reg_w_path_norm * unique_cg_val[fiber.w];
            reg_dL_din1[fiber.x] += dL_dOuter * reg_in2[fiber.y];
            reg_dL_din2[path_info2.y+fiber.y] += dL_dOuter * reg_in1[fiber.x];
            reg_uvuv[fiber.z] += reg_in1[fiber.x] * reg_in2[fiber.y] * unique_cg_val[fiber.w];
        }

        // // // debugging
        // for (int i =0, dldo_idx = g_t_dldo_start + path_info1.x + threadIdx.x*path_info2.x; i<path_info2.x; i++, dldo_idx++){
        //     mem_debug[dldo_idx] = shmem_warp_pos_start;
        // }

        // mult dL_dO with uvuv
        float reg_dL_dw = 0.0;
        for(int k_idx = 0; k_idx<path_info2.x; k_idx++){
            reg_dL_dw += reg_dldo[k_idx] * reg_uvuv[k_idx] * path_weight[path_idx];
        }
        // store dL_dW
        // per path store
        mem_dL_dW[weight_pos] = reg_dL_dw;
    }

    // store dL_dA first in shared mem
    for(int i =0, shmem_idx = threadIdx.x*i_val; i<i_val; i++, shmem_idx++){
        shmem_dldo[shmem_idx] = reg_dL_din1[i];
    }
    __syncwarp();
    // store dL_dA in main mem
    for(int shmem_idx = threadIdx.x, in1_idx = global_t_batch_idx*in1_size + in1_start+threadIdx.x; in1_idx < global_t_batch_idx*in1_size+ in1_end; in1_idx+=WARPSIZE, shmem_idx+=WARPSIZE) {
        mem_dL_din1[in1_idx] = shmem_dldo[shmem_idx];
    }

    // warp shuffle reduce
    for (int i = 0; i < in2_size; i++) {
        float sum = reg_dL_din2[i];
        for (int offset = 1; offset < WARPSIZE; offset *= 2) {
            sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);
        }
        reg_dL_din2[i] = sum;
    }

    // load all in2 from main mem 
    // larger by the number of in1 accumulate (len(i_in1))
    const int g_dl_din2 = global_t_batch_idx*path_cnt*in2_size + target_in1*in2_size;

    for(int i = threadIdx.x, in2_idx = g_dl_din2 + threadIdx.x; in2_idx < g_dl_din2+in2_size; i+=WARPSIZE, in2_idx+=WARPSIZE) {
        mem_dL_din2[in2_idx] = reg_dL_din2[i];
    }

}


void bwd_sptp_linear_cuda_v1(
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
    size_t per_block_batch
    ){

    // TODO: not transposed (z, mul, ir)
    const auto batch_size = in1.size(0);
    const auto in1_size = in1.size(1);
    const auto in2_size = in2.size(1);
    const auto out_size = mem_dL_dO.size(1);
    const auto weight_size = weight.size(1);

    dim3 grid(path_cnt, batch_size/per_block_batch);
    dim3 block(WARPSIZE, per_block_batch);

    // setup constant memory 
    cudaMemcpyToSymbol(in1_idxing, t_in1_idxing.data<int>(), at::numel(t_in1_idxing)*sizeof(int)); // int , MAX_IN1_IR_CNT
    cudaMemcpyToSymbol(in1_ival, t_in1_ival.data<int>(),  at::numel(t_in1_ival)*sizeof(int)); // int , MAX_IN1_IR_CNT
    cudaMemcpyToSymbol(in1_related_path_idx, t_in1_related_path_idx.data<int>(), at::numel(t_in1_related_path_idx)*sizeof(int)); // int  , MAX_IN1_IR_CNT
    
    cudaMemcpyToSymbol(path_array1, t_path_array1.data<u_short>(), at::numel(t_path_array1)*sizeof(u_short) ); // ushort2, MAX_NUM_PATH
    cudaMemcpyToSymbol(path_array2, t_path_array2.data<u_char>(), at::numel(t_path_array2)*sizeof(u_char)); // uchar4, MAX_NUM_PAT
    cudaMemcpyToSymbol(per_path_fiber_start, t_per_path_fiber_start.data<u_short>(), at::numel(t_per_path_fiber_start)*sizeof(u_short)); // ushort2, MAX_NUM_PATH
    cudaMemcpyToSymbol(path_weight, t_path_weight.data<float>(), at::numel(t_path_weight)*sizeof(float)); // float, MAX_NUM_PATH
    cudaMemcpyToSymbol(per_path_weight_pos, t_per_path_weight_pos.data<int>(), at::numel(t_per_path_weight_pos)*sizeof(int)); // int , MAX_NUM_PATH

    cudaMemcpyToSymbol(fiber_array, t_fiber_array.data<u_char>(), at::numel(t_fiber_array)*sizeof(u_char)); // u_char4), MAX_U_FIBER_CNT 
    cudaMemcpyToSymbol(unique_cg_val, t_unique_cg_val.data<float>(), at::numel(t_unique_cg_val)*sizeof(float) ); // float , MAX_U_CG_VAL_CNT

    const int shared_memory_bytes = sizeof(float) * (WARPSIZE * per_block_batch * MAX_IR + in2_size*per_block_batch);

    // int carveout = 100;
    // CHECK_CUDA_ERROR(cudaFuncSetAttribute(
    //     sptp_all_forward_kernel_v1<float>,
    //     cudaFuncAttributePreferredSharedMemoryCarveout, carveout));

    CHECK_CUDA_ERROR(cudaFuncSetAttribute(
        bwd_sptp_lienar_kernel_v1<float>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_bytes));

    bwd_sptp_lienar_kernel_v1<float><<<grid, block, shared_memory_bytes>>>(
        in1.data<float>(),
        in2.data<float>(),
        weight.data<float>(),
        mem_dL_dO.data<float>(),
        
        mem_dL_din1.data<float>(),
        mem_dL_din2.data<float>(),
        mem_dL_dW.data<float>(),
        mem_debug.data<float>(),

        batch_size,
        out_size,
        weight_size,
        in1_size,
        in2_size,
        path_cnt
        );
}
    