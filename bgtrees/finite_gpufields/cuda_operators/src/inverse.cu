#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "inverse.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include <cmath>

using namespace tensorflow;

namespace myfunctor {

    typedef Eigen::GpuDevice GPUDevice;

    template <typename T>
        __global__ void InverseCudaKernel(const T* x, T* out) {
		// Copied from: https://github.com/GDeLaurentis/linac-dev/blob/master/linac/row_reduce.cu
        
            const int n = blockIdx.x * blockDim.x + threadIdx.x;

            T quotient, old_old_r, old_old_s, old_old_t;
            T b = PMOD;

            T old_r = x[n];
            T r = b;
            T old_s = 1;
            T s = 0;
            T old_t = 0;
            T t = 1;

            while (r != 0) {
                quotient = old_r / r;
                old_old_r = old_r;
                old_r = r;
                r = old_old_r - quotient * old_r;
                old_old_s = old_s;
                old_s = s;
                s = old_old_s - quotient * old_s;
                old_old_t = old_t;
                old_t = t;
                t = old_old_t - quotient * old_t;
            }

            s = old_s;

            if (s > 0) {
                out[n] = s;
            } else {
                out[n] = s + PMOD;
            }
        }

    template <typename T>
        struct InverseFunctor<GPUDevice, T> {
            void operator()(const GPUDevice& d, const int bs, const T* x, T* out) {
                int thread_per_block = 1;
                int block_count = static_cast<int>(std::ceil(static_cast<double>(bs) / thread_per_block));
                InverseCudaKernel<T>
                    <<<block_count, thread_per_block, 0, d.stream()>>>(x, out);
            }
        };
 

    // Explicitly instantiate functors for the types of OpKernels registered.

    template struct InverseFunctor<GPUDevice, int64>;
    template struct InverseFunctor<GPUDevice, int32>;

}  // end namespace functor

#endif  // GOOGLE_CUDA
