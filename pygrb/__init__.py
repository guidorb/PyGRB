"""Top-level package for PyGRB."""

__author__ = """Guido Roberts-Borsani"""
__email__ = 'g.robertsborsani@ucl.ac.uk'
__version__ = '0.1.0'

from . import general_functions
from . import analysis_functions
from . import spectral_functions
from . import luminosity_functions
from . import student_functions

try:
    from . import dja_functions
except ImportError:
    pass

try:
    from . import jewels_fit
except ImportError:
    pass

try:
    from . import jwst_reduction
except (ImportError, SyntaxError):
    pass

from .general_functions import (
    load_jewels as jewels,
    load_jewels,
    style_axes,
    find_nearest,
    fnu_to_flam,
    flam_to_fnu,
    flux_to_AB,
    AB_to_flux,
    photo_from_filter,
    get_filter_info,
)

from .analysis_functions import (
    bin_1d_spec,
    stack_spectra,
    generate_line_mask,
)

from .spectral_functions import (
    LineFitting,
    inverse_variance_mean,
    inverse_variance_mean_sigma_clip,
    gaussian_smooth_spectrum,
    stack_1d,
)

from .luminosity_functions import (
    MUV_to_LUV,
    LUV_to_MUV,
    Schechter,
    DoublePower,
    Donnan23,
    Donnan24,
    Harikane23_Schechter,
    Harikane23_DPL,
)
