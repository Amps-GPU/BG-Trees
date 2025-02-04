import numpy
import sympy

from syngular import Ring, Ideal, QRing

from .metric_and_vertices import η


def dDimPhaseSpaceQRing(m, D):
    """Quotient ring representing the on-shell, momentum conserving, m-point, D-dimensional phase space point"""
    momenta = numpy.array([numpy.array([sympy.symbols(f"p{i}_{j}") for j in range(D)]) for i in range(1, m + 1)])
    on_shell_relations = [momentum @ η(D) @ momentum for momentum in momenta]
    momentum_conservation = sum(momenta).tolist()
    r = Ring('0', tuple(momenta.flatten().tolist()), 'dp')
    i = Ideal(r, on_shell_relations + momentum_conservation)
    q = QRing(r, i)
    return q


def random_phase_space_point(m, D, field, as_dict=False, seed=None):
    """Return a random m-point D-dimensional phase space point in the given field"""
    qRing = dDimPhaseSpaceQRing(m, D)
    point = qRing.random_point(field=field, seed=seed)
    if as_dict:
        return point
    lDMoms = numpy.array([[point[f"p{i}_{j}"] for j in range(D)] for i in range(1, m + 1)])
    return lDMoms


def μ2(momD, d=None):
    """
    The 4D mass of a massless D momentum (d=None).
    For μ^2_d = k^2_4 + ... + k^2_d-1
    """
    D = momD.shape[0]
    if d is None:
        d = D
    return - momD[4:d] @ η(D)[4:d, 4:d] @ momD[4:d]


def momflat(momD, momχ):
    """Massless (`flat`) projection of 4D massive momentum onto a reference direction momχ."""
    D = momD.shape[0]
    return momD[:4] - μ2(momD) * momχ / (2 * momχ @ η(D)[:4, :4] @ momD[:4])
