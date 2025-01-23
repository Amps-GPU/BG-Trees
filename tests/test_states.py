import pytest
import numpy
import lips

from syngular import Field
from pyadic import rationalise

from bgtrees.metric_and_verticies import η
from bgtrees.states import all_states
from bgtrees.phase_space import random_phase_space_point, μ2

lips.spinor_convention = "asymmetric"


@pytest.mark.parametrize("D", (4, 5, 6, 7, 8, 9, 10))
def test_compleness_relation_in_Ddims(D):
    """Tests completeness relations as defined in eq. 14 and 15 of arXiv:250X.XXXXX."""

    field = Field("finite field", 2 ** 31 - 19, 1)
    momD = random_phase_space_point(2, D, field)[0]  # D-mom
    momχ = random_phase_space_point(2, 4, field)[0]  # 4D ref. vector
    xsi = numpy.block([momD[:4], -momD[4:]])  # D-dim ref. vector

    states = all_states(momD, momχ, field)
    states_sum = sum([numpy.outer(state, state_conj) for state, state_conj in zip(states[0], states[1])])

    if D == 4:
        metric = - (
            states_sum -
            (numpy.outer(momχ, momD) + numpy.outer(momD, momχ)) / (momD @ η(D) @ momχ)
        )
        assert numpy.all(numpy.vectorize(rationalise)(metric) == numpy.diag([1, -1, -1, -1] + [-1] * (D - 4)))
    elif D > 4:
        # as a pair of massive states in D=4 and D=(D-4) - see eq. 14 of arXiv:250X.XXXXX
        metric = - (
            states_sum -
            numpy.block([[(numpy.outer(momD[:4], momD[:4]) / μ2(momD)), numpy.zeros((4, D - 4), dtype=int)],
                         [numpy.zeros((D - 4, 4), dtype=int), -numpy.outer(momD[4:], momD[4:]) / μ2(momD)]])
        )
        assert numpy.all(numpy.vectorize(rationalise)(metric) == numpy.diag([1, -1, -1, -1] + [-1] * (D - 4)))
        # massless in D-dim - see eq. 13 of arXiv:250X.XXXXX
        metric2 = - (
            states_sum -
            (numpy.outer(momD, xsi) + numpy.outer(xsi, momD)) / (momD @ η(D) @ xsi)
        )
        assert numpy.all(numpy.vectorize(rationalise)(metric2) == numpy.diag([1, -1, -1, -1] + [-1] * (D - 4)))
    else:
        raise ValueError("D not in allowed range.")
