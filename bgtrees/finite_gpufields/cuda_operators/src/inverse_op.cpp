#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("Inverse")
.Attr("T: {int32, int64}")
.Input("x: T")
.Output("out: T")
.SetShapeFn([](shape_inference::InferenceContext* c) {
    c->set_output(0, c->input(0));
    return tensorflow::Status();
});
