import pytest
import numpy
import lips

from syngular import Field
from pyadic import rationalise

from bgtrees.metric_and_verticies import η
from bgtrees.states import ε1, ε2, ε3, ε3c, ε4, ε4c, εxs, εxcs
from bgtrees.phase_space import random_phase_space_point, μ2

lips.spinor_convention = "asymmetric"


@pytest.mark.parametrize("D", (4, 5, 6, 7, 8, 9, 10))
def test_compleness_relation_in_Ddims(D):

    print(D, type(D))

    field = Field("finite field", 2 ** 31 - 19, 1)
    momD = random_phase_space_point(2, D, field)[0]
    momχ = random_phase_space_point(2, 4, field)[0]

    print(momD)
    print(momχ)

    e1 = ε1(momD, momχ, field)
    e2 = ε2(momD, momχ, field)
    e1c, e2c = e2, e1
    if D >= 5:
        e3, e3c = ε3(momD, momχ, field), ε3c(momD, momχ, field)
    if D >= 6:
        e4, e4c = ε4(momD), ε4c(momD)
    if D >= 7:
        exs, excs = [εxs(momD, x) for x in range(5, D - 1)], [εxcs(momD, x) for x in range(5, D - 1)]

    if D == 4:
        metric = - (
            (numpy.outer(e1, e2) + numpy.outer(e2, e1)) -
            (numpy.outer(momχ, momD) + numpy.outer(momD, momχ)) / (momD @ η(D) @ momχ)
        )
    if D == 5:
        metric = - (
            (numpy.outer(e1, e1c) + numpy.outer(e2, e2c) + numpy.outer(e3, e3c)) -
            numpy.block([[(numpy.outer(momD[:4], momD[:4]) / μ2(momD)), numpy.zeros((4, D - 4), dtype=int)],
                         [numpy.zeros((D - 4, 4), dtype=int), -numpy.outer(momD[4:], momD[4:]) / μ2(momD)]])
        )
    if D == 6:
        metric = - (
            (numpy.outer(e1, e1c) + numpy.outer(e2, e2c) + numpy.outer(e3, e3c) + numpy.outer(e4, e4c)) -
            numpy.block([[(numpy.outer(momD[:4], momD[:4]) / μ2(momD)), numpy.zeros((4, D - 4), dtype=int)],
                        [numpy.zeros((D - 4, 4), dtype=int), -numpy.outer(momD[4:], momD[4:]) / μ2(momD)]])
        )
    if D >= 7:
        metric = - (
            (numpy.outer(e1, e1c) + numpy.outer(e2, e2c) + numpy.outer(e3, e3c) + numpy.outer(e4, e4c) +
             sum([numpy.outer(exs[i], excs[i]) for i in range(D - 6)])) -
            numpy.block([[(numpy.outer(momD[:4], momD[:4]) / μ2(momD)), numpy.zeros((4, D - 4), dtype=int)],
                        [numpy.zeros((D - 4, 4), dtype=int), -numpy.outer(momD[4:], momD[4:]) / μ2(momD)]])
        )

    assert numpy.all(numpy.vectorize(rationalise)(metric) == numpy.diag([1, -1, -1, -1] + [-1] * (D - 4)))
