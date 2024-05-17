#include "inverse.h"
#include <stdio.h>

// CPU specialization of actual computation.
template <typename T>
struct myfunctor::InverseFunctor<CPUDevice, T> {
    void operator()(const CPUDevice &d, const int bs, const T *x, T *out) {

		// Copied from: https://github.com/GDeLaurentis/linac-dev/blob/master/linac/row_reduce.cu
        
        for (int n = 0; n < bs; n++) {
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

            out[n] = s;
            if (s < 0) {
                out[n] += PMOD;
            }
        }
    }
};

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T>
InverseOp<Device, T>::InverseOp(OpKernelConstruction* context) : OpKernel(context) {}

template <typename Device, typename T>
void InverseOp<Device, T>::Compute(OpKernelContext* context) {

	const Tensor& input_x = context->input(0);
	const auto input_data = input_x.flat<T>();
    const auto batch_s =input_x.NumElements();

    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_x.shape(), &output_tensor));
	auto output_data = output_tensor->flat<T>();

	// Submit to the appropiate device
    myfunctor::InverseFunctor<Device, T>()(context->eigen_device<Device>(), batch_s, input_data.data(), output_data.data());
}

// Register the CPU kernels.
#define REGISTER_CPU(T)                                          \
        REGISTER_KERNEL_BUILDER(                                       \
                Name("Inverse").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
                InverseOp<CPUDevice, T>);

REGISTER_CPU(int64);
REGISTER_CPU(int32);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
// Define the EIGEN_USE_GPU and instantiate the template
#define EIGEN_USE_GPU

#define REGISTER_GPU(T)                                          \
        extern template struct myfunctor::InverseFunctor<GPUDevice, T>;           \
        REGISTER_KERNEL_BUILDER(                                       \
                Name("Inverse").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
                InverseOp<GPUDevice, T>);

REGISTER_GPU(int64);
REGISTER_GPU(int32);
#endif  // GOOGLE_CUDA
