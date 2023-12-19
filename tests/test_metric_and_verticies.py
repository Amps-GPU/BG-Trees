import tensorflow
import numpy

from bgtrees.settings import settings
from bgtrees.metric_and_verticies import MinkowskiMetric, V4g, V3g


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def test_MinkowskiMetric_cpu_vs_gpu():
    settings.use_gpu = True
    metric_gpu = MinkowskiMetric(5)
    settings.use_gpu = False
    metric_cpu = MinkowskiMetric(5)
    # dtypes
    assert metric_gpu.dtype is tensorflow.int64
    assert not metric_gpu.dtype is numpy.dtype('int64')
    assert not metric_cpu.dtype is tensorflow.int64
    assert metric_cpu.dtype is numpy.dtype('int64')
    # values
    assert numpy.isclose(metric_gpu, metric_cpu).all()


def test_V3g_cpu_vs_gpu():
    p1, p2 = numpy.array([1, 1, 0, 0, 0]), numpy.array([1, -1, 0, 0, 0])
    settings.use_gpu = True
    V3g_gpu = V3g(p1, p2)
    settings.use_gpu = False
    V3g_cpu = V3g(p1, p2)
    assert numpy.isclose(V3g_cpu, V3g_gpu).all()
