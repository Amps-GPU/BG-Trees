// kernel_example.h taken from tensorflow custom op repository
#ifndef KERNEL_dot_product_H_
#define KERNEL_dot_product_H_

#define PMOD 2147483629

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace myfunctor {

	template <typename Device, typename T>
	struct InverseFunctor {
		void operator()(const Device& d, 
                const int bs,
				const T* x, // input tensor X
				T* out      // output tensor
		);
	};

}  // namespace functor
    
template <typename Device, typename T>
class InverseOp: public OpKernel {
public:
	explicit InverseOp(OpKernelConstruction* context);
	void Compute(OpKernelContext* context) override;
};

#endif //KERNEL_dot_product_H_
