target_lib=dot_product.so

TF_CFLAGS=`python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))' 2> /dev/null`
TF_LFLAGS=`python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))' 2>/dev/null`

CXX=g++
CXFLAGS=-std=c++17 -shared -fPIC -O3 -ffast-math #-D DEBUGVERBOSE=1
# warnings are disabled because most of them are coming from tensorflow itself...
NCCFLAGS=-std=c++17 -I$(subst bin/nvcc,include,$(shell which nvcc)) $(KERNEL_DEF) -x cu -Xcompiler -fPIC --disable-warnings --expt-relaxed-constexpr 
NCC:=nvcc
NCCLIB:=$(subst bin/nvcc,lib, $(shell which nvcc))
KERNEL_DEF=-D GOOGLE_CUDA=1 -lcudart -L$(NCCLIB) -L$(NCCLIB)64

# If there's no cuda compiler in the system remove %.cuo from the %.so rule
# and clean the KERNEL_DEF line so that the cuda kernel is not used

CXFLAGS+=$(KERNEL_DEF) 

.PHONY: run clean

run: $(target_lib)
	@CUDA_VISIBLE_DEVICES=0 python py_dotproduct.py

run_cpu: $(target_lib)
	@CUDA_VISIBLE_DEVICES="" python py_dotproduct.py

%.so: src/%_op.cpp src/%.cuo src/%.cpp
	@echo "[$(CXX)] Compiling and linking operation"
	@$(CXX) $(CXFLAGS) $(KERNEL) -o $@ $^ $(TF_CFLAGS) $(TF_LFLAGS)

src/%.cuo: src/%.cu
	@echo "[$(NCC)] Compiling cuda kernel..."
	@$(NCC) $(NCCFLAGS) -c -o $@ $^ $(TF_CFLAGS)

.PRECIOUS: src/%.cuo src/%.o %.so

clean:
	rm -f *.so src/*. src/*.cuo
