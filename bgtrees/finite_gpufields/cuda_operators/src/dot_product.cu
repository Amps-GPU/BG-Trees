#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "dot_product.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include <cmath>

using namespace tensorflow;

namespace myfunctor {

    typedef Eigen::GpuDevice GPUDevice;

    // Dot Produt
    template <typename T, bool singleBatch>
        __global__ void DotProductCudaKernel(const int bs, const int o1, const int o2, const int size_i, const T* x, const T* y, T* out) {
            /*
                o1 and o2 correspond to the size of the first and last outer indices
                size_i is the size of the uncontracted index
            */
            const int n = blockIdx.x * blockDim.x + threadIdx.x;
            if (n >= bs) return;

            const int b_o = n*o1*o2;
            const int b_x = n*o1*size_i;

            int b_y;
            if constexpr(singleBatch) {
                b_y = 0;
            } else {
                b_y = n * o2 * size_i;
            }

            for (int i = 0; i < o1; i++) {
                for (int j = 0; j < o2; j++) {
                    T res = 0;
                    for (int k = 0; k < size_i; k++) {
                        const T tmp = (x[i*size_i + k + b_x] * y[k*o2 + j + b_y]) % PMOD;
                        res = (res + tmp) % PMOD;
                    }

                    out[i*o2 + j + b_o] = res;
                }
            }
        }

    template <typename T, bool singleBatch>
        struct DotProductFunctor<GPUDevice, T, singleBatch> {
            void operator()(const GPUDevice& d, const int bs, const int o1, const int o2, const int size_i, const T* x, const T* y, T* out) {
                /* Run in the given GPU device (d) the operation below
                    
                    bs: batch size, size of the uncontracted index, the vectorization will be along this index
                    o1 and o2 correspond to the size of the first and last outer indices
                    size_i is the size of the uncontracted index
                */ 
                int thread_per_block = 1000;
                int block_count = static_cast<int>(std::ceil(static_cast<double>(bs) / thread_per_block));
                DotProductCudaKernel<T, singleBatch>
                    <<<block_count, thread_per_block, 0, d.stream()>>>(bs, o1, o2, size_i, x, y, out);
            }
        };

    // Explicitly instantiate functors for the types of OpKernels registered.

    // Double batched
    template struct DotProductFunctor<GPUDevice, int64, false>;
    template struct DotProductFunctor<GPUDevice, int32, false>;

    // Single batched
    template struct DotProductFunctor<GPUDevice, int64, true>;
    template struct DotProductFunctor<GPUDevice, int32, true>;

}  // end namespace functor

#endif  // GOOGLE_CUDA
