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

#define MAX_K 16
#define WARPSIZE 32

template <typename scalar_t>
__global__ void sptp_all_forward_kernel_linear_v1(
    const float* __restrict__ in1,
    const float* __restrict__ in2,
    float* __restrict__ out,
    const float* __restrict__ uni_w3j,
    const float* __restrict__ weight,
    const float* __restrict__ path_weight,

    const u_int16_t* __restrict__ per_blk_w_start_idx,
    const u_int16_t* __restrict__ per_blk_w_end_idx,
    const u_int16_t* __restrict__ per_blk_w_precnt,
    const u_int16_t* __restrict__ per_blk_k_dim,

    const u_int16_t* __restrict__ per_blk_u_in1_idx_range,
    const u_int16_t* __restrict__ per_blk_u_in2_idx_range,
    const u_int16_t* __restrict__ per_blk_fiber_start_range,

    const u_int16_t* __restrict__ blk_u_in1_idx,
    const u_int16_t* __restrict__ blk_u_in2_idx,
    
    const u_int16_t* __restrict__ pattern_len_array,
    const u_int16_t* __restrict__ stride_mul_array,
    const u_int16_t* __restrict__ rem_cum_idx_array,
    const u_int16_t* __restrict__ rem_cumval_array,
    const u_int16_t* __restrict__ fiber_cnt_array,

    const u_int8_t* __restrict__ per_fiber_local_idx,
    
    const size_t batch_size,
    const size_t max_u_in1_dim,
    const size_t max_u_in2_dim,
    const size_t uni_cg_cnt,
    const size_t out_size
    )
    {
    extern __shared__ scalar_t shmem[];

    // block x is z / warp size
    // block y is total linear set / (warpcnt)

    // x is warp dimension going for different batch
    // y target different linear set

    const int b_batch_idx = blockIdx.x;
    const int b_set_idx = blockIdx.y;

    const int t_l_batch_idx = threadIdx.x;
    const int t_l_set_idx = threadIdx.y;
    const int concurrent_batch = blockDim.x;

    const int thread_num = threadIdx.y * blockDim.x + threadIdx.x;
    const int total_thread_cnt = blockDim.x * blockDim.y;

    const int global_t_batch_idx = b_batch_idx * concurrent_batch + t_l_batch_idx;
    const int global_t_set_idx = b_set_idx * blockDim.y + t_l_set_idx;

    scalar_t* shmem_in1 = shmem;
    scalar_t* shmem_in2 = shmem_in1 + max_u_in1_dim * concurrent_batch;
    scalar_t* shmem_cg = shmem_in2 + max_u_in2_dim * concurrent_batch;
    scalar_t* shmem_w = shmem_cg + uni_cg_cnt;
    scalar_t* shmem_pw = shmem_w + WARPSIZE * concurrent_batch;
    scalar_t* shmem_out = shmem_pw + WARPSIZE;
    u_int8_t* uint8_shmem_fiber = (u_int8_t*) (shmem_out + WARPSIZE*WARPSIZE);

    const int out_start = b_set_idx * WARPSIZE; // multiple of 32
    const int out_end = (b_set_idx+1) * WARPSIZE; // multiple of 32
    
    const int b_fiber_start = per_blk_fiber_start_range[b_set_idx];
    const int b_fiber_end = per_blk_fiber_start_range[b_set_idx+1];    

    // load cg
    for(int i=thread_num; i<uni_cg_cnt; i+=total_thread_cnt){
        shmem_cg[i] = uni_w3j[i];
    }
    // load path_weight
    for(int i=out_start+thread_num; i<out_end; i+=total_thread_cnt){
        shmem_pw[i-out_start] = path_weight[i];
    }
    
    const int w_start = per_blk_w_start_idx[b_set_idx];
    const int w_end = per_blk_w_end_idx[b_set_idx];
    const int w_previous = per_blk_w_precnt[b_set_idx];
    const int l_k_dim = per_blk_k_dim[b_set_idx];
    const int my_w_pos = (t_l_set_idx+w_previous) / l_k_dim;

    // load weight (special case where v = 1)
    for(int i=w_start+t_l_set_idx; i<w_end; i+= blockDim.y){
        shmem_w[(i-w_start)*concurrent_batch + t_l_batch_idx] = weight[i*batch_size + global_t_batch_idx];
    }
    // load per_fiber_local_idx
    for(int i=b_fiber_start*4+thread_num; i<b_fiber_end*4; i+=total_thread_cnt){
        uint8_shmem_fiber[i-b_fiber_start*4] = per_fiber_local_idx[i];
    }
    uchar4 *uint8_shmem_fiber_v  = reinterpret_cast<uchar4 *>(uint8_shmem_fiber);

    const int u_in1_start = per_blk_u_in1_idx_range[b_set_idx];
    const int u_in1_end = per_blk_u_in1_idx_range[b_set_idx+1];

    const int u_in2_start = per_blk_u_in2_idx_range[b_set_idx];
    const int u_in2_end = per_blk_u_in2_idx_range[b_set_idx+1];

    // different uni_in1_end for each warp blockDim.y => different set
    // load unique_in1
    for(int i=t_l_set_idx; i< u_in1_end-u_in1_start; i+=blockDim.y){
        shmem_in1[i*concurrent_batch+t_l_batch_idx] = in1[blk_u_in1_idx[i+u_in1_start]*batch_size+ global_t_batch_idx];
    }
    // load unique in2
    for(int i=t_l_set_idx; i< u_in2_end-u_in2_start; i+=blockDim.y){
        shmem_in2[i*concurrent_batch+t_l_batch_idx] = in2[blk_u_in2_idx[i+u_in2_start]*batch_size+ global_t_batch_idx];
    }
    
    //
    const int pattern_length = pattern_len_array[b_set_idx];
    const int stride_mul = stride_mul_array[b_set_idx]; // get stride for this warp
    const int rem_cum_idx = rem_cum_idx_array[b_set_idx]; // get array for this warp
    const int stride = t_l_set_idx / pattern_length; // get pos of this thread in the warp for fiber 
    const int rem = t_l_set_idx % pattern_length;  // get pos of this trhead in the warp for fiber

    // get it from pattern
    const int w_fiber_start = stride * stride_mul + rem_cumval_array[rem_cum_idx+rem];
    const int w_fiber_end = w_fiber_start + fiber_cnt_array[rem_cum_idx+rem];

    __syncthreads();
    //debugging
    // out[global_t_batch_idx * out_size + global_t_set_idx] = shmem_w[t_l_set_idx * concurrent_batch + t_l_batch_idx];

    float Cout = 0.0;
    for(int fiber_idx = w_fiber_start; fiber_idx<w_fiber_end; fiber_idx++ ){
        uchar4 fiber = uint8_shmem_fiber_v[fiber_idx];
        // Cout += shmem_in1[0]; break;
        // Cout += shmem_in1[fiber.x*concurrent_batch+t_l_batch_idx]; break;
        // Cout += shmem_cg[fiber.z]; break;
        Cout += shmem_in1[fiber.x*concurrent_batch+t_l_batch_idx] * shmem_in2[fiber.y*concurrent_batch+t_l_batch_idx] * shmem_cg[fiber.z];
    }

    // mutliply weight
    Cout *= shmem_pw[t_l_set_idx] * shmem_w[my_w_pos * concurrent_batch + t_l_batch_idx];
    // // write to shmem 

    shmem_out[t_l_set_idx*concurrent_batch+(t_l_batch_idx+t_l_set_idx)%WARPSIZE] = Cout;

    __syncthreads();
    
    // // write to maim mem
    const int target_g_set_idx = b_set_idx * blockDim.y + t_l_batch_idx;
    const int target_g_batch_idx = b_batch_idx * concurrent_batch + t_l_set_idx;

    out[target_g_batch_idx * out_size + target_g_set_idx] = shmem_out[t_l_batch_idx*WARPSIZE+(t_l_batch_idx+t_l_set_idx)%WARPSIZE];

}


void sptp_all_forward_linear_v1_cuda(
    torch::Tensor in1, 
    torch::Tensor in2,
    torch::Tensor out, 
    torch::Tensor uni_w3j, 
    torch::Tensor weight, 
    torch::Tensor path_weight,

    torch::Tensor per_blk_w_start_idx,
    torch::Tensor per_blk_w_end_idx,
    torch::Tensor per_blk_w_precnt,
    torch::Tensor per_blk_k_dim,

    torch::Tensor per_blk_u_in1_idx_range,
    torch::Tensor per_blk_u_in2_idx_range,
    torch::Tensor per_blk_fiber_start_range,
    torch::Tensor blk_u_in1_idx,
    torch::Tensor blk_u_in2_idx,

    torch::Tensor pattern_len_array,
    torch::Tensor stride_mul_array,
    torch::Tensor rem_cum_idx_array,
    torch::Tensor rem_cumval_array,
    torch::Tensor fiber_cnt_array,

    torch::Tensor per_fiber_local_idx,

    size_t max_unique_in1_size, 
    size_t max_unique_in2_size, 
    size_t max_fiber_cnt,
    size_t uni_cg_cnt){

    // TODO: all transposed
    const auto batch_size = out.size(0);
    const auto out_size = out.size(1);
    const auto weight_size = weight.size(0);
    const int num_thread = WARPSIZE;

    dim3 grid(batch_size/WARPSIZE, out_size/WARPSIZE);
    dim3 block(WARPSIZE, WARPSIZE);

    const int shared_memory_bytes = sizeof(float) * 
    ((max_unique_in1_size + max_unique_in2_size + WARPSIZE+WARPSIZE) * num_thread + WARPSIZE + uni_cg_cnt) + max_fiber_cnt * 4 * sizeof(u_int8_t);

    // int carveout = 100;
    // CHECK_CUDA_ERROR(cudaFuncSetAttribute(
    //     sptp_all_forward_kernel_v1<float>,
    //     cudaFuncAttributePreferredSharedMemoryCarveout, carveout));

    CHECK_CUDA_ERROR(cudaFuncSetAttribute(
        sptp_all_forward_kernel_linear_v1<float>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_bytes));

    sptp_all_forward_kernel_linear_v1<float><<<grid, block, shared_memory_bytes>>>(
        in1.data<float>(),
        in2.data<float>(),
        out.data<float>(),
        uni_w3j.data<float>(),
        weight.data<float>(),
        path_weight.data<float>(),

        per_blk_w_start_idx.data<u_int16_t>(),
        per_blk_w_end_idx.data<u_int16_t>(),
        per_blk_w_precnt.data<u_int16_t>(),
        per_blk_k_dim.data<u_int16_t>(),

        per_blk_u_in1_idx_range.data<u_int16_t>(),
        per_blk_u_in2_idx_range.data<u_int16_t>(),
        per_blk_fiber_start_range.data<u_int16_t>(),
        blk_u_in1_idx.data<u_int16_t>(),
        blk_u_in2_idx.data<u_int16_t>(),
        
        pattern_len_array.data<u_int16_t>(),
        stride_mul_array.data<u_int16_t>(),
        rem_cum_idx_array.data<u_int16_t>(),
        rem_cumval_array.data<u_int16_t>(),
        fiber_cnt_array.data<u_int16_t>(),

        per_fiber_local_idx.data<u_int8_t>(),

        batch_size,
        max_unique_in1_size,
        max_unique_in2_size,
        uni_cg_cnt,
        out_size
        );
}
    