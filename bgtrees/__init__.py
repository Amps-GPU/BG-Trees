"""
    Public-facing functions

        compute_current_j_mu:
            compute current per event given arrays of momenta and polarization

        generate_batch_phase_space
            generate a batch momentum-conserving phase space point for a given field
"""

import lips
import numpy as np
import tensorflow as tf

from ._version import __version__  # noqa
from .currents import J_μ, another_j
from .finite_gpufields.finite_fields_tf import FiniteField
from .phase_space import random_phase_space_point
from .settings import settings
from .states import ε1, ε2, ε3, ε4


def generate_batch_points(multiplicity=4, dimension=6, batch_size=3, field_type="ff", helconf="ppmm"):
    """Generate a batch of random momentum conserving phase space points.
    Requires syngular.

    Returns a tuple containing momenta and polarization for each phase space point,
    both arrays of shape (events, multiplicity, dimension)
    """
    # TODO: pass seed through
    try:
        from syngular import Field
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("Please install `syngular` to generate phase space points.") from e

    if len(helconf) != multiplicity:
        raise ValueError(f"Please, make sure that multiplicity ({multiplicity}) and helconf ({helconf}) are consistent")

    if field_type.lower() in ("ff", "finitefield"):
        field = Field("finite field", settings.p, 1)
        settings.dtype = np.int64
        lips.spinor_convention = "asymmetric"

        def convert(xinput):
            # Make it into a container
            # TODO: check whether there's a benefit on the container in CPU as well, otherwise keep the modP type
            return FiniteField(np.array(xinput).astype(int), settings.p)

    elif field_type.lower() in ("mpc", "float"):
        field = Field("mpc", 0, 300)
        # settings.dtype = np.float64

        def convert(xinput):
            return np.array(xinput)

    else:
        raise ValueError(f"Field type not understood: {field_type}")

    lmoms = []
    lpols = []

    reference_vector = random_phase_space_point(2, 4, field, seed=74)[0]
    for _ in range(batch_size):
        momenta = np.array(random_phase_space_point(multiplicity, dimension, field))

        tmp = []
        for idx, hel in enumerate(helconf):
            if hel in ("1", "m"):
                polarization_function = ε1
            elif hel in ("2", "p"):
                polarization_function = ε2
            elif hel == 3:
                polarization_function = ε3
            elif hel == 4:
                polarization_function = ε4
            else:
                raise Exception(f"Polarization not understood {hel}.")

            pol = polarization_function(momenta[idx], reference_vector, field)
            tmp.append(np.block([pol]))

        lmoms.append(momenta)
        lpols.append(tmp)

    lmoms = convert(lmoms)
    lpols = convert(lpols)
    return lmoms, lpols


def compute_current_j_mu(lmoms, lpols, put_propagator=True):
    """
    Recursive vectorized current builder. End of recursion is polarization tensors.

    The momenta and polarization arrays can be any number of phase space points in
    any multiplicity or dimensionality.

    The shape of the input arrays must be (events, multiplicity, dimensions)
    """
    # Check whether we are working with a Finite Field TF container
    if isinstance(lmoms, FiniteField) or isinstance(lpols, FiniteField):
        # Safety check: they are both finite fields
        if (tm := type(lmoms)) != (tp := type(lpols)):
            raise TypeError(
                f"If either momenta or polarization are Finite Fields both must be. Momenta: {tm}, Polarizations: {tp}"
            )
        # If we have a Finite Field container and they are both finite field, we are good to go
        return another_j(lmoms, lpols, put_propagator=put_propagator)

    # Depending on the type of the input we can use different versions of J_mu
    if isinstance(lmoms, np.ndarray):
        if (tm := type(lmoms.flat[0])) != (tp := type(lpols.flat[0])):
            raise TypeError(f"The type of momenta ({tm}) and polarizations ({tp}) differ.")

        return J_μ(lmoms, lpols, put_propagator, einsum=np.einsum)
    elif isinstance(lmoms, tf.Tensor):
        return J_μ(lmoms, lpols, put_propagator, einsum=tf.einsum)
    else:
        raise TypeError(
            f"Type {type(lmoms)} not recognized or supported. You might be lucky accessing directly the low-level interface?"
        )
