import numpy
import tensorflow

from bgtrees.metric_and_verticies import MinkowskiMetric, V3g, new_V3g
from bgtrees.settings import settings

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def test_MinkowskiMetric_cpu_vs_gpu():
    settings.use_gpu = True
    metric_gpu = MinkowskiMetric(5)
    settings.use_gpu = False
    metric_cpu = MinkowskiMetric(5)
    # dtypes
    assert isinstance(metric_gpu.dtype, tensorflow.dtypes.DType)
    assert not isinstance(metric_gpu.dtype, numpy.dtype)
    assert not isinstance(metric_cpu.dtype, tensorflow.dtypes.DType)
    assert isinstance(metric_cpu.dtype, numpy.dtype)
    # values
    assert numpy.isclose(metric_gpu, metric_cpu).all()


def test_V3g_cpu_vs_gpu():
    lp1, lp2 = numpy.array([[1, 1, 0, 0, 0]]), numpy.array([[1, -1, 0, 0, 0]])
    settings.use_gpu = True
    V3g_gpu = new_V3g(lp1, lp2)
    settings.use_gpu = False
    V3g_cpu = V3g(lp1, lp2)
    # dtype
    assert isinstance(V3g_gpu.dtype, tensorflow.dtypes.DType)
    assert not isinstance(V3g_gpu.dtype, numpy.dtype)
    assert not isinstance(V3g_cpu.dtype, tensorflow.dtypes.DType)
    assert isinstance(V3g_cpu.dtype, numpy.dtype)
    # values
    assert numpy.isclose(V3g_cpu, V3g_gpu).all()
