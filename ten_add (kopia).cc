#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

#include <immintrin.h>
#include <iostream>

using namespace tensorflow;

REGISTER_OP("TenAdd")
	.Input("input_tensor_a: float")
	.Input("input_tensor_b: float")
	.Output("zeroed: float")
	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
		c->set_output(0, c->input(0));
		return Status::OK();
	});

class TenAddOp : public OpKernel {
public:
	explicit TenAddOp(OpKernelConstruction* context) : OpKernel(context) {}

	void Compute(OpKernelContext* context) override {
		const Tensor& input_tensor = context->input(0);
		auto input = input_tensor.flat<float>();

		Tensor* output_tensor = NULL;
		OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));
		auto output_flat = output_tensor->flat<float>();

		const int N = input.size();
		const float* raw_input = input.data();
		float* raw_output = output_flat.data();
		__m256 one = _mm256_set_ps(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
		__m256 value;
		__m256 wyn;
		
		for(int i = 0; i < N; i+=8, raw_input+=8, raw_output+=8){
			value = _mm256_loadu_ps((float const*)raw_input);
			wyn = _mm256_add_ps(one, value);
			_mm256_storeu_ps(raw_output, wyn);
		}
		
	}
};

REGISTER_KERNEL_BUILDER(Name("TenAdd").Device(DEVICE_CPU), TenAddOp);
