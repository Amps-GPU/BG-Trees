"""
Definition of states (polarisation vectors) in D dimensions.
See eq.~11 to 18 of arXiv:250X.XXXXX.
"""

from lips import Particle, Particles
import numpy

from .phase_space import momflat, μ2


def ε1(momD, momχ, field):
    """Corresponds to plus helicity in D=4."""
    D, momFlat = len(momD), momflat(momD, momχ)
    ε1 = Particle(
        Particles([Particle(momFlat, field=field), Particle(momχ, field=field)], field=field, fix_mom_cons=False)(
            "(-|2]⟨1|)/([1|2])"
        ),
        field=field,
    )
    return numpy.append(ε1.four_mom, (field(0),) * (D - 4))


def ε2(momD, momχ, field):
    """Corresponds to minus helicity in D=4"""
    D, momFlat = len(momD), momflat(momD, momχ)
    ε2 = Particle(
        Particles([Particle(momFlat, field=field), Particle(momχ, field=field)], field=field, fix_mom_cons=False)(
            "2(|1]⟨2|)/(⟨1|2⟩)"
        ),
        field=field,
    )
    return numpy.append(ε2.four_mom, (field(0),) * (D - 4))


def ε3(momD, momχ, field):
    D, momFlat = len(momD), momflat(momD, momχ)
    if D < 5:
        raise ValueError(f"Not enough dimensions for ε3, need D>=5, was given D={D}.")
    ε3 = Particle(
        Particles(
            [Particle(momFlat, field=field), Particle(momχ, field=field), Particle(momD[:4], field=field)],
            field=field,
            fix_mom_cons=False,
            internal_masses={"μ2": μ2(momD)},
        )("(|1]⟨1|)-μ2*|2]⟨2|/(⟨2|3|2])"),
        field=field,
    )
    return numpy.append(ε3.four_mom, (field(0),) * (D - 4))


def ε3c(momD, momχ, field):
    return ε3(momD, momχ, field) / μ2(momD)


def ε4(momD):
    D = len(momD)
    if D < 6:
        raise ValueError(f"Not enough dimensions for ε4, need D>=6, was given D={D}.")
    ε4 = numpy.block(
        [numpy.array([0] * 4), numpy.array([[1, 0], [0, -1]]) @ momD[4:6][::-1], numpy.array([0] * (D - 6))]
    ) / μ2(momD, 6)
    return ε4


def ε4c(momD):
    return ε4(momD) * μ2(momD, 6)


def εxs(momD, x):
    """D >= 7 polarization states. Needs x in {5, ..., D - 2}."""
    D = len(momD)
    if D < 7:
        raise ValueError(f"Not enough dimensions for εx, need D>=7, was given D={D}.")
    return (
        numpy.block(
            [
                numpy.array([0] * 4),
                numpy.array([momD[j] * momD[x + 1] for j in range(4, x + 1)] + [-μ2(momD, x + 1)] + [0] * (D - x - 2)),
            ]
        )
        / μ2(momD, x + 1)
        / μ2(momD, x + 2)
    )


def εxcs(momD, x):
    """'Conjugate polarization state."""
    return εxs(momD, x) * μ2(momD, x + 1) * μ2(momD, x + 2)


def all_states(momD, momχ, field):
    """Returns a complete set of states and their conjugates."""

    D = len(momD)

    if D < 4:
        raise ValueError("Expected at least 4 dimensions.")

    e1 = e2c = ε1(momD, momχ, field)
    e2 = e1c = ε2(momD, momχ, field)

    states = [e1, e2]
    states_conj = [e1c, e2c]

    if D == 4:
        return states, states_conj

    e3, e3c = ε3(momD, momχ, field), ε3c(momD, momχ, field)
    states += [e3]
    states_conj += [e3c]

    if D == 5:
        return states, states_conj

    e4, e4c = ε4(momD), ε4c(momD)
    states += [e4]
    states_conj += [e4c]

    if D == 5:
        return states, states_conj

    exs, excs = [εxs(momD, x) for x in range(5, D - 1)], [εxcs(momD, x) for x in range(5, D - 1)]
    states += exs
    states_conj += excs

    return states, states_conj
