from jaxfun.typing import InnerKind as InnerKind, MeshKind as MeshKind

from . import (
    Chebyshev as Chebyshev,
    ChebyshevU as ChebyshevU,
    Fourier as Fourier,
    Jacobi as Jacobi,
    Legendre as Legendre,
    Ultraspherical as Ultraspherical,
    orthogonal as orthogonal,
)
from .arguments import (
    Array as Array,
    JAXFunction as JAXFunction,
    PhysicalArray as PhysicalArray,
    TestFunction as TestFunction,
    TrialFunction as TrialFunction,
)
from .cartesianproductspace import (
    CartesianProduct as CartesianProduct,
    CartesianProductSpace as CartesianProductSpace,
    CartesianTensorProductSpace as CartesianTensorProductSpace,
    VectorTensorProductSpace as VectorTensorProductSpace,
)
from .composite import Composite as Composite, DirectSum as DirectSum
from .functionspace import FunctionSpace as FunctionSpace
from .inner import (
    Project as Project,
    inner as inner,
    inner_items as inner_items,
    integrate as integrate,
)
from .tensorproductspace import (
    CoupledSpace as CoupledSpace,
    DirectSumTPS as DirectSumTPS,
    K_over_K2 as K_over_K2,
    TensorProduct as TensorProduct,
    TensorProductSpace as TensorProductSpace,
)
