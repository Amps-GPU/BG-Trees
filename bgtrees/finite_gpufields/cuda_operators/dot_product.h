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

	template <typename Device, typename T, bool singleBatch>
	struct DotProductFunctor {
		// Dot product A_{rji}B^{jk}_{r} = C_{rj}^{k}
		// ie, \sum_{i} A_{rji}*B_{rjk} = C_{rjk}
		// the index r is the batch index and is the index along which the vectorization happen
        //
        // When the boolean singleBatch, defined at compile time, is set to True,
        // the computation is instead A_{rji}B^{jk} = C_{rj}^{k}
        // i.e., only the first array is batched, and all elements are contracted with the second
		void operator()(const Device& d, 
				const int batch_size, // size of the vectorized dimensioned
				const int size_j, // size of the uncontracted first index of input X
				const int size_k, // size of the uncontracted last index of input Y
				const int size_i, // contracted index
				const T* x, // input tensor X
				const T* y, // input tensor Y
				T* out      // output tensor
		);
	};

}  // namespace functor
    
template <typename Device, typename T, bool singleBatch>
class DotProductOp: public OpKernel {
public:
	explicit DotProductOp(OpKernelConstruction* context);
	void Compute(OpKernelContext* context) override;
};

#endif //KERNEL_dot_product_H_
