// This file defines the abstract operation
// This includes any attributes (for instance, this operation will act on int32
// or int64) the inputs (which are tensors, of any shape, of the type defined by
// T) and the outputs

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "dot_product.h"

REGISTER_OP("DotProduct")
.Attr("T: {int32, int64}")
.Input("x: T")
.Input("y: T")
.Output("out: T")
.SetShapeFn([](shape_inference::InferenceContext* c) {
        // The output is defined according to the input arrrays
        // for the time being assume they are both rank 2 arrays
        // so the output is shape_x[1],shape_y[2]
        // Note that the first dimension of every array is always a batch-dimension
        const auto x = c->input(0);
        const auto y = c->input(1);

        const auto output_shape = c->MakeShape({
                c->Dim(x, 0),
                c->Dim(x,1),
                c->Dim(y,2),
                });

        c->set_output(0, output_shape);
        return tensorflow::Status();
        });
