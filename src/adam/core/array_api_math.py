from adam.core.spatial_math import ArrayLike, SpatialMath, ArrayLikeFactory
import array_api_compat as aac
from typing import Any, Callable, Optional
from dataclasses import dataclass
from types import ModuleType


@dataclass(frozen=True)
class ArraySpec:
    xp: ModuleType  # array API namespace (compat-wrapped if needed)
    dtype: Optional[Any]  # xp.float32, torch.float32, jnp.float32, etc.
    device: Optional[Any]  # xp device object (torch device, jax device, "cpu", ...)


def spec_from_reference(ref: Any) -> ArraySpec:
    # Force compat namespace when available (useful for PyTorch/JAX).
    # JAX doesn't have an array-api-compat wrapper, so use use_compat=False for JAX
    try:
        xp = aac.array_namespace(ref, use_compat=True)
    except ValueError as e:
        if "JAX does not have an array-api-compat wrapper" in str(e):
            xp = aac.array_namespace(ref, use_compat=False)
        else:
            raise
    dtype = getattr(ref, "dtype", None)
    # aac.device(x) provides spec-like device, including a CPU device for NumPy.
    try:
        device = aac.device(ref)
    except Exception:
        device = getattr(ref, "device", None)
    return ArraySpec(xp=xp, dtype=dtype, device=device)


def xp_getter(*xs: Any):
    return aac.array_namespace(*xs)  # use_compat=True?


@dataclass
class ArrayAPILike(ArrayLike):
    """Generic Array-API-style wrapper used by NumPy/JAX/Torch backends."""

    array: Any

    def __getitem__(self, idx) -> "ArrayAPILike":
        return self.__class__(self.array[idx])

    @property
    def shape(self):
        return self.array.shape

    def reshape(self, *args):
        xp = xp_getter(self.array)
        return xp.reshape(self.array, *args)

    @property
    def T(self) -> "ArrayAPILike":
        if getattr(self.array, "ndim", 0) == 0:
            return self.__class__(self.array)
        xp = xp_getter(self.array)
        return self.__class__(
            xp.swapaxes(self.array, 0, -1)  # if self.array.ndim != 0 else self.array
        )

    def __matmul__(self, other):
        xp = xp_getter(self.array, other.array)
        return self.__class__(xp.matmul(self.array, other.array))

    def __rmatmul__(self, other) -> "ArrayAPILike":
        xp = xp_getter(self.array, other.array)
        return self.__class__(xp.matmul(other.array, self.array))

    def __mul__(self, other) -> "ArrayAPILike":
        xp = xp_getter(self.array, other.array)
        return self.__class__(xp.multiply(self.array, other.array))

    def __rmul__(self, other) -> "ArrayAPILike":
        xp = xp_getter(self.array, other.array)
        return self.__class__(xp.multiply(self.array, other.array))

    def __truediv__(self, other) -> "ArrayAPILike":
        xp = xp_getter(self.array, other.array)
        return self.__class__(xp.divide(self.array, other.array))

    def __add__(self, other) -> "ArrayAPILike":
        xp = xp_getter(self.array, other.array)
        return self.__class__(xp.add(self.array, other.array))

    def __radd__(self, other) -> "ArrayAPILike":
        xp = xp_getter(self.array, other.array)
        return self.__class__(other.array + self.array)

    def __sub__(self, other) -> "ArrayAPILike":
        xp = xp_getter(self.array, other.array)
        return self.__class__(self.array - other.array)

    def __rsub__(self, other) -> "ArrayAPILike":
        xp = xp_getter(self.array, other.array)
        return self.__class__(xp.squeeze(other.array) - xp.squeeze(self.array))

    def __neg__(self) -> "ArrayAPILike":
        return self.__class__(-self.array)

    @property
    def ndim(self):
        return self.array.ndim


class ArrayAPIFactory(ArrayLikeFactory):
    """
    Generic factory. Give it (a) a Like class and (b) an xp namespace
    (array_api_compat.* if available; otherwise the library module).
    """

    def __init__(self, like_cls, xp, *, dtype=None, device=None):
        self._like = like_cls
        self._xp = xp
        self._dtype = dtype
        self._device = device

    def zeros(self, *shape) -> ArrayAPILike:
        # Handle tuple concatenation like H.shape[:-2] + (1, 4)
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            final_shape = shape[0]
        else:
            final_shape = shape

        x = self._xp.zeros(final_shape, dtype=self._dtype, device=self._device)
        return self._like(x)

    def eye(self, *shape) -> ArrayAPILike:
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            x = shape[0][-1]
            batch = shape[0][:-1]
        else:
            batch = shape[:-1]
            x = shape[-1]
        return self._like(
            self._xp.eye(x, dtype=self._dtype, device=self._device)
            if batch is None
            else self._xp.broadcast_to(
                self._xp.eye(x, dtype=self._dtype, device=self._device), batch + (x, x)
            )
        )

    def asarray(self, x) -> ArrayAPILike:
        # preserve the gradient if x is a torch tensor (check if it has "requires_grad_" attribute)
        # it could be moved to the torch-like class, but maybe here is more visible
        if getattr(x, "requires_grad_", False):
            return self._like(x.to(device=self._device, dtype=self._dtype))
        return self._like(self._xp.asarray(x, dtype=self._dtype, device=self._device))

    def zeros_like(self, x: ArrayAPILike) -> ArrayAPILike:
        return self._like(self._xp.zeros_like(x.array, dtype=x.array.dtype))

    def ones_like(self, x: ArrayAPILike) -> ArrayAPILike:
        return self._like(self._xp.ones_like(x.array, dtype=x.array.dtype))

    def tile(self, x: ArrayAPILike, reps: tuple) -> ArrayAPILike:
        return self._like(self._xp.tile(x.array, reps))

    def broadcast_to(self, x: ArrayAPILike, shape: tuple) -> ArrayAPILike:
        return self._like(self._xp.broadcast_to(x.array, shape))


class ArrayAPISpatialMath(SpatialMath):
    """A drop-in SpatialMath that implements sin/cos/outer/concat/skew with the Array API.

    Works for NumPy, PyTorch, and JAX; CasADi should keep its own subclass.
    """

    def __init__(self, factory, xp_getter: Callable[..., Any] = xp_getter):
        super().__init__(factory)
        self._xp_getter = xp_getter

    def _xp(self, *xs: Any):
        return self._xp_getter(*xs)

    def sin(self, x):
        xp = self._xp(x.array)
        x = x.array
        return self.factory.asarray(xp.sin(x))

    def cos(self, x):
        xp = self._xp(x.array)
        x = x.array
        return self.factory.asarray(xp.cos(x))

    def skew(self, x):
        xp = self._xp(x.array)
        a = x.array
        # if x is batched (shape (B, 3, 1)), remove the last dimension
        if a.ndim >= 2 and a.shape[-1] == 1:
            a = a[..., 0]
        x0, x1, x2 = a[..., 0], a[..., 1], a[..., 2]
        z = x0 * 0
        row0 = xp.stack([z, -x2, x1], axis=-1)
        row1 = xp.stack([x2, z, -x0], axis=-1)
        row2 = xp.stack([-x1, x0, z], axis=-1)
        return self.factory.asarray(xp.stack([row0, row1, row2], axis=-2))  # (...,3,3)

    def outer(self, x, y):
        xp = self._xp(x.array, y.array)
        a = x.array
        b = y.array
        # normalize to (...,3)
        if a.ndim >= 2 and a.shape[-2] == 3 and a.shape[-1] == 1:
            a = a[..., :, 0]
        if b.ndim >= 2 and b.shape[-2] == 3 and b.shape[-1] == 1:
            b = b[..., :, 0]
        # (...,3,1) @ (...,1,3) -> (...,3,3)
        A = a[..., :, None]
        B = b[..., None, :]
        return self.factory.asarray(xp.matmul(A, B))

    def vertcat(self, *x):
        xp = self._xp(*[xi.array for xi in x])
        return self.factory.asarray(xp.vstack([xi.array for xi in x]))

    def horzcat(self, *x):
        xp = self._xp(*[xi.array for xi in x])
        return self.factory.asarray(xp.hstack([xi.array for xi in x]))

    def stack(self, x, axis=0):
        xp = self._xp(x[0].array)
        return self.factory.asarray(xp.stack([xi.array for xi in x], axis=axis))

    def concatenate(self, x, axis=0):
        xp = self._xp(x[0].array)
        return self.factory.asarray(xp.concatenate([xi.array for xi in x], axis=axis))

    def swapaxes(self, x: ArrayAPILike, axis1: int, axis2: int) -> ArrayAPILike:
        xp = self._xp(x.array)
        return self.factory.asarray(xp.swapaxes(x.array, axis1, axis2))

    def expand_dims(self, x: ArrayAPILike, axis: int) -> ArrayAPILike:
        xp = self._xp(x.array)
        return self.factory.asarray(xp.expand_dims(x.array, axis=axis))

    def transpose(self, x: ArrayAPILike, dims: tuple) -> ArrayAPILike:
        xp = self._xp(x.array)
        return self.factory.asarray(xp.permute_dims(x.array, dims))

    def inv(self, x: ArrayAPILike) -> ArrayAPILike:
        xp = self._xp(x.array)
        return self.factory.asarray(xp.linalg.inv(x.array))

    def mtimes(self, A: ArrayAPILike, B: ArrayAPILike) -> ArrayAPILike:
        xp = self._xp(A.array, B.array)
        return self.factory.asarray(xp.matmul(A.array, B.array))

    def solve(self, A: ArrayAPILike, B: ArrayAPILike) -> ArrayAPILike:
        xp = self._xp(A.array, B.array)
        a, b = A.array, B.array
        # A right-hand side with one dim fewer than A is a (batch of) vector(s).
        if b.ndim == a.ndim - 1:
            return self.factory.asarray(xp.linalg.solve(a, b[..., None])[..., 0])
        return self.factory.asarray(xp.linalg.solve(a, b))
