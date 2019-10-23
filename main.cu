#include <iostream>

#include <vector>
#include <cstdio>
#include <exception>

/** macro to throw a runtime error */
#define THROW(fmt, ...)                                                    \
  do {                                                                     \
    std::string msg;                                                       \
    char errMsg[2048];                                                     \
    std::sprintf(errMsg, "Exception occured! file=%s line=%d: ", __FILE__, \
                 __LINE__);                                                \
    msg += errMsg;                                                         \
    std::sprintf(errMsg, fmt, ##__VA_ARGS__);                              \
    msg += errMsg;                                                         \
    throw std::runtime_error(msg);                                         \
  } while (0)

/** macro to check for a conditional and assert on failure */
#define ASSERT(check, fmt, ...)              \
  do {                                       \
    if (!(check)) THROW(fmt, ##__VA_ARGS__); \
  } while (0)


/** check for cuda runtime API errors and assert accordingly */
#define CUDA_CHECK(call)                                                \
  do {                                                                  \
    cudaError_t status = call;                                          \
    ASSERT(status == cudaSuccess, "FAIL: call='%s'. Reason:%s\n", #call, \
           cudaGetErrorString(status));                                 \
  } while (0)

__global__ void init_xy(double* x, double* y, double a, double b, int N) {
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int idx = bid*blockDim.x + tid;
  if(idx < N) {
    x[idx] = a;
    y[idx] = b;
  }
}


__global__ void saxpy(const double* x, const double* y, const double a, const int N, double* z) {
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int idx = bid*blockDim.x + tid;
  if(idx < N) {
    z[idx] = a*x[idx] + y[idx];
    // printf("z[%d]=%f * %f + %f = %f\n", idx, a, x[idx], y[idx], z[idx]);
  }
}

int checkResults(double* d_z, double a, const int N) {
  std::vector<double> h_z(N);
  CUDA_CHECK(cudaMemcpy(h_z.data(), d_z, sizeof(double)*N, cudaMemcpyDeviceToHost));

  // check
  int i=0;
  for(auto zi : h_z) {
    if(zi != a*1 + 2) {
      std::printf("ERROR: z[%d]=%f != %fn", i, zi, a*1 + 2);
      return -1;
    }
    i++;
  }
  return 0;
}

int main() {
  const int N = 100;
  double* d_x;
  double* d_y;
  double* d_z;
  CUDA_CHECK(cudaMalloc(&d_x, sizeof(double)*N));
  CUDA_CHECK(cudaMalloc(&d_y, sizeof(double)*N));
  CUDA_CHECK(cudaMalloc(&d_z, sizeof(double)*N));
  

  const int threads_per_block = 64;
  const int num_blocks = ceil((double)N/threads_per_block);
  // printf("Num blocks=%d\n", num_blocks);

  init_xy<<<num_blocks, threads_per_block>>>(d_x, d_y, 1.0, 2.0, N);
  CUDA_CHECK(cudaGetLastError());

  double a = 42.0;
  saxpy<<<num_blocks, threads_per_block>>>(d_x, d_y, a, N, d_z);
  CUDA_CHECK(cudaGetLastError());
  
  return checkResults(d_z, a, N);
}
