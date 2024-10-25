import itertools
import pytest
import numpy

from lips import Particle
from syngular import Field

from bgtrees.metric_and_verticies import η
from bgtrees.phase_space import random_phase_space_point, momflat


# Generate all combinations of the following parameters
fields = [
    Field("finite field", 2 ** 31 - 19, 1),
    Field("mpc", 0, 300),
    Field("padic", 2 ** 31 - 19, 11)
]
m_values = [5, 6]
D_values = [6, 7, 8]
param_combinations = list(itertools.product(fields, m_values, D_values))


@pytest.mark.parametrize("field, m, D", param_combinations)
def test_Ddim_phase_space(field, m, D):
    lDMoms = random_phase_space_point(m, D, field)
    # check on-shell-ness
    assert all([abs(DMom @ η(D) @ DMom) <= field.tollerance for DMom in lDMoms])
    # check momentum conservation
    assert all(abs(sum(lDMoms)) <= field.tollerance)


@pytest.mark.parametrize("field, m, D", param_combinations)
def test_flat_moms(field, m, D):
    lDMoms = random_phase_space_point(m, D, field)
    χ = Particle(field=field)
    momχ = χ.four_mom
    lMomsFlat = numpy.array([momflat(momD, momχ) for momD in lDMoms])
    # phase space point is not massless in D=4
    assert all([abs(momD[:4] @ η(D)[:4, :4] @ momD[:4]) >= field.tollerance for momD in lDMoms])
    # flattened phase space is massless in D=4
    assert all([abs(momFlat @ η(D)[:4, :4] @ momFlat) <= field.tollerance for momFlat in lMomsFlat])
