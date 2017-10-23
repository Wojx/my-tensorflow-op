#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

#include <immintrin.h>
#include <iostream>

using namespace tensorflow;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("TenAdd")
	.Input("input_tensor_a: float")
	.Input("input_tensor_b: float")
	.Output("sumed_tensor: float")
	.SetShapeFn([](InferenceContext* c) {
		ShapeHandle cur = c->input(0);
		for (int i = 0; i < c->num_inputs(); ++i) {
        		TF_RETURN_WITH_CONTEXT_IF_ERROR(c->Merge(c->input(i), cur, &cur),
							"From merging shape ", i,
							" with other shapes.");
		}		
		c->set_output(0, c->input(0));
		
		return Status::OK();
	});

class TenAddOp : public OpKernel {
public:
	explicit TenAddOp(OpKernelConstruction* context) : OpKernel(context) {}

	void Compute(OpKernelContext* context) override {
		const Tensor& first_input_tensor = context->input(0);
		const Tensor& second_input_tensor = context->input(1);
		
		auto first_input = first_input_tensor.flat<float>();
		auto second_input = second_input_tensor.flat<float>();

		Tensor* output_tensor = NULL;
		OP_REQUIRES_OK(context, context->allocate_output(0, first_input_tensor.shape(), &output_tensor));
		auto output_flat = output_tensor->flat<float>();

		const int N = first_input.size();
		const float* raw_first_input = first_input.data();
		const float* raw_second_input = second_input.data();

		float* raw_output = output_flat.data();
		__m256 value1;
		__m256 value2;
		__m256 wyn;
		if(N >= 8){
			for(int i = 0; i < N/8; ++i, raw_first_input+=8, raw_second_input+=8, raw_output+=8){
				value1 = _mm256_loadu_ps(raw_first_input);
				value2 = _mm256_loadu_ps(raw_second_input);
				wyn = _mm256_add_ps(value1, value2);
				_mm256_storeu_ps(raw_output, wyn);
			}
			for(int i = N - N % 8; i < N; ++i){
				output_flat(i) = first_input(i) + second_input(i);
			}	
		}else{
			for(int i = 0; i < N; ++i){
				output_flat(i) = first_input(i) + second_input(i);
			}
		}		
	}
};

REGISTER_KERNEL_BUILDER(Name("TenAdd").Device(DEVICE_CPU), TenAddOp);
