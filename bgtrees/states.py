import numpy

from lips import Particle, Particles

from .phase_space import momflat, μ2


def ε1(momD, momχ, field):
    D, momFlat = len(momD), momflat(momD, momχ)
    ε1 = Particle(Particles([Particle(momFlat, field=field), Particle(momχ, field=field)],
                            field=field, fix_mom_cons=False)("(-|2]⟨1|)/([1|2])"), field=field)
    return numpy.append(ε1.four_mom, (field(0), ) * (D - 4))


def ε2(momD, momχ, field):
    D, momFlat = len(momD), momflat(momD, momχ)
    ε2 = Particle(Particles([Particle(momFlat, field=field), Particle(momχ, field=field)],
                            field=field, fix_mom_cons=False)("2(|1]⟨2|)/(⟨1|2⟩)"), field=field)
    return numpy.append(ε2.four_mom, (field(0), ) * (D - 4))


def ε3(momD, momχ, field):
    D, momFlat = len(momD), momflat(momD, momχ)
    if D < 5:
        raise ValueError(f"Not enough dimensions for ε3, need D>=5, was given D={D}.")
    ε3 = Particle(Particles([Particle(momFlat, field=field), Particle(momχ, field=field),
                             Particle(momD[:4], field=field)], field=field, fix_mom_cons=False,
                            internal_masses={'μ2': μ2(momD)})("(|1]⟨1|)-μ2*|2]⟨2|/(⟨2|3|2])"), field=field)
    return numpy.append(ε3.four_mom, (field(0), ) * (D - 4))


def ε3c(momD, momχ, field):
    return ε3(momD, momχ, field) / μ2(momD)


def ε4(momD, momχ, field):
    D = len(momD)
    if D < 6:
        raise ValueError(f"Not enough dimensions for ε4, need D>=6, was given D={D}.")
    ε4 = numpy.block([numpy.array([0, ] * 4),
                      numpy.array([[1, 0], [0, -1]]) @ momD[4:6][::-1],
                      numpy.array([0, ] * (D - 6))]) / μ2(momD, 6)
    return ε4


def ε4c(momD, momχ, field):
    return ε4(momD, momχ, field) * μ2(momD, 6)


def εxs(momD, x):
    """D >= 7 polarization states. Needs x in {5, ..., D - 2}."""
    D = len(momD)
    if D < 7:
        raise ValueError(f"Not enough dimensions for εx, need D>=7, was given D={D}.")
    return numpy.block([numpy.array([0, ] * 4),
                        numpy.array([momD[j] * momD[x + 1] for j in range(4, x + 1)] +
                                    [-μ2(momD, x + 1)] + [0, ] * (D - x - 2))
                        ]) / μ2(momD, x + 1) / μ2(momD, x + 2)


def εxcs(momD, x):
    """'Conjugate polarization state."""
    return εxs(momD, x) * μ2(momD, x + 1) * μ2(momD, x + 2)
