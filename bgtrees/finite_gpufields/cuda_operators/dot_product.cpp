#include "tensorflow/core/framework/tensor.h"
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif // GOOGLE_CUDA
#include "tensorflow/core/framework/op_kernel.h"
#include "dot_product.h"

#include <stdio.h>

namespace tensorflow {

    typedef Eigen::ThreadPoolDevice CPUDevice;
    typedef Eigen::GpuDevice GPUDevice;

    namespace functor {
        // CPU specialization of actual computation.
        template <typename T>
            struct DotProductFunctor<CPUDevice, T> {
                void operator()(const CPUDevice& d, const int bs, const int o1, const int o2, const int size_i, const T* x, const T* y, T* out) {

                    for (int n = 0; n < bs; n++) {

                        const int b_o = n*o1*o2;
                        const int b_x = n*o1*size_i;
                        const int b_y = n*o2*size_i;

                        for (int i = 0; i < o1; i++) {
                            for (int j = 0; j < o2; j++) {
                                T res = 0;
                                for (int k = 0; k < size_i; k++) {
                                    const int tmp = (x[i*size_i + k + b_x] * y[k*o2 + j + b_y]) % PMOD;
                                    res = (res + tmp) % PMOD;
                                }
                                out[i*o2 + j + b_o] = 0*res;
                            }
                        }

                    }


                }
            };

        // OpKernel definition.
        // template parameter <T> is the datatype of the tensors.
        template <typename Device, typename T>
            class DotProductOp : public OpKernel {
                public:
                    explicit DotProductOp(OpKernelConstruction* context) : OpKernel(context) {}

                    void Compute(OpKernelContext* context) override {

                        // Grab the input
                        const Tensor& input_x = context->input(0);
                        const Tensor& input_y = context->input(1);

                        // And the shapes and dimensions
                        const auto& shape_x = input_x.shape();
                        const auto& shape_y = input_y.shape();
                        
                        const auto batch_size = shape_x.dim_size(0);
                        const auto outer_x = shape_x.dim_size(1);
                        const auto outer_y = shape_y.dim_size(2);
                        const auto inner_s = shape_y.dim_size(1);

                        // (optional) check that everything is ok
                        //DCHECK_EQ(run_size, input_x.shape().dim_size(1));
                        
                        // Prepare the shape of hte output tensor
                        TensorShape output_shape;
                        output_shape.AddDim(batch_size);
                        output_shape.AddDim(outer_x);
                        output_shape.AddDim(outer_y);

                        // Create the output tensor
                        Tensor* output_tensor = NULL;
                        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape,&output_tensor));

                        DotProductFunctor<Device, T>()(
                                context->eigen_device<Device>(),
                                static_cast<int>(batch_size),
                                static_cast<int>(outer_x),
                                static_cast<int>(outer_y),
                                static_cast<int>(inner_s),
                                input_x.flat<T>().data(),
                                input_y.flat<T>().data(),
                                output_tensor->flat<T>().data());
                    }
            };

        // Register the CPU kernels.
#define REGISTER_CPU(T)                                          \
        REGISTER_KERNEL_BUILDER(                                       \
                Name("DotProduct").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
                DotProductOp<CPUDevice, T>);
        REGISTER_CPU(int64);
        REGISTER_CPU(int32);

        // Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                          \
        extern template struct DotProductFunctor<GPUDevice, T>;           \
        REGISTER_KERNEL_BUILDER(                                       \
                Name("DotProduct").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
                DotProductOp<GPUDevice, T>);
        REGISTER_GPU(int64);
        REGISTER_GPU(int32);
#endif  // GOOGLE_CUDA
    }
}  // namespace tensorflow
