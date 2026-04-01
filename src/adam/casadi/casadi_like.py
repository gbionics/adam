# Copyright (C) Istituto Italiano di Tecnologia (IIT). All rights reserved.

from dataclasses import dataclass
from typing import Union, Sequence

import casadi as cs
import numpy.typing as npt

from adam.core.spatial_math import (
    ArrayLike,
    ArrayLikeFactory,
    SpatialMath as _SpatialMath,
)


@dataclass
class CasadiLike(ArrayLike):
    """Wrapper class for CasADi SX/DM with ArrayLike ops."""

    array: Union[cs.SX, cs.DM]

    def __matmul__(self, other: "CasadiLike") -> "CasadiLike":
        return CasadiLike(cs.mtimes(self.array, other.array))

    def __rmatmul__(self, other: "CasadiLike") -> "CasadiLike":
        return CasadiLike(cs.mtimes(other.array, self.array))

    def __mul__(self, other: "CasadiLike") -> "CasadiLike":
        return CasadiLike(self.array * other.array)

    def __rmul__(self, other: "CasadiLike") -> "CasadiLike":
        return CasadiLike(other.array * self.array)

    def __truediv__(self, other: "CasadiLike") -> "CasadiLike":
        return CasadiLike(self.array / other.array)

    def __add__(self, other: "CasadiLike") -> "CasadiLike":
        a, b = self.array, other.array
        sa, sb = a.shape, b.shape

        # Scalars always broadcast in CasADi
        if sa == sb or (sa == (1, 1)) or (sb == (1, 1)):
            return CasadiLike(a + b)

        # If one is a vector and the other is the same vector transposed, align to self
        # column (n,1) + row (1,n) is undefined for elementwise add; we only allow when shapes match after T
        if sa == (sb[1], sb[0]) and (1 in sb):
            return CasadiLike(a + b.T)

        # If both are vectors with same length but different orientation, align to self
        if sa[1] == 1 and sb[0] == 1 and sa[0] == sb[1]:  # self col, other row
            return CasadiLike(a + b.T)
        if sa[0] == 1 and sb[1] == 1 and sa[1] == sb[0]:  # self row, other col
            return CasadiLike(a + b.T)

        raise ValueError(f"Shape mismatch for add: {sa} + {sb}")

    def __radd__(self, other: "CasadiLike") -> "CasadiLike":
        return CasadiLike(other.array).__add__(self)

    def __sub__(self, other: "CasadiLike") -> "CasadiLike":
        return CasadiLike(self.array - other.array)

    def __rsub__(self, other: "CasadiLike") -> "CasadiLike":
        return CasadiLike(other.array - self.array)

    def __neg__(self) -> "CasadiLike":
        return CasadiLike(-self.array)

    # --- indexing / shape / transpose ---
    def __getitem__(self, idx) -> "CasadiLike":
        # CasADi is 2-D; strip ellipsis/None and keep the remaining 2 indices max.
        if idx is Ellipsis:
            return self

        if isinstance(idx, tuple):
            # remove Ellipsis
            idx = tuple(i for i in idx if i is not Ellipsis)
            # remove None (newaxis); CasADi doesn't support >2D, so just ignore
            idx = tuple(i for i in idx if i is not None)
            if not idx:
                return self
            if len(idx) == 1:
                idx = idx[0]
            elif len(idx) > 2:
                # Keep only last two indices (row, col)
                idx = idx[-2:]

        return CasadiLike(self.array[idx])

    @property
    def shape(self) -> tuple[int, ...]:
        return self.array.shape

    @property
    def ndim(self) -> int:
        return len(self.array.shape)

    @property
    def T(self) -> "CasadiLike":
        return CasadiLike(self.array.T)


class CasadiLikeFactory(ArrayLikeFactory):
    """ArrayLikeFactory for CasADi. Drops batch dims (>2) since CasADi is 2-D only."""

    def __init__(self, xp: Union[cs.SX, cs.DM, None] = None):

        self._xp = cs.SX if xp is None else xp

        # else:
        #     super().__init__(CasadiLike, xp)

    def zeros(self, *x: npt.ArrayLike) -> CasadiLike:
        # Accept zeros((..batch.., r, c)) or zeros(r, c)
        if len(x) == 1 and isinstance(x[0], (tuple, list)):
            shp = tuple(x[0])
        else:
            shp = tuple(x)
        if len(shp) > 2:
            shp = shp[-2:]
        if len(shp) == 0:
            shp = (1,)
        return CasadiLike(self._xp.zeros(*shp))

    def eye(self, x: npt.ArrayLike) -> CasadiLike:
        # Accept eye(n) or eye((..batch.., n))
        n = x[-1] if isinstance(x, (tuple, list)) else x
        return CasadiLike(self._xp.eye(int(n)))

    def asarray(self, x) -> CasadiLike:
        """
        Convert input to a CasadiLike array.

        This method handles various input types and converts them to CasadiLike objects
        using appropriate CasADi operations for concatenation and array construction.

        Args:
            x: Input to convert. Can be:
            - Empty list: Returns empty CasadiLike array
            - List of CasADi objects (cs.SX, cs.DM): Horizontally concatenated
            - List of lists/tuples: Creates 2D array with vertical and horizontal concatenation
            - Numbers or lists of numbers: Direct conversion to CasadiLike

        Returns:
            CasadiLike: A CasadiLike object wrapping the converted input.

        Examples:
            - Empty list [] -> CasadiLike with empty array
            - [sx1, sx2] -> CasadiLike with horizontally concatenated SX objects
            - [[1, 2], [3, 4]] -> CasadiLike with 2x2 matrix
            - 5 or [1, 2, 3] -> CasadiLike with direct conversion
        """
        # Handle empty list case
        if isinstance(x, list):
            if not x:
                return CasadiLike(self._xp([]))
            # List contains CasADi objects - concatenate horizontally
            if any(isinstance(it, (cs.SX, cs.DM)) for it in x):
                return CasadiLike(self._xp(cs.vertcat(*x)))
            # List of lists/tuples - create 2D array with vertical and horizontal concatenation
            if all(isinstance(it, (list, tuple)) for it in x):
                return CasadiLike(
                    self._xp(
                        cs.vertcat(
                            *[cs.horzcat(*[self._xp(e) for e in it]) for it in x]
                        )
                    )
                )
        # Direct conversion for numbers or lists of numbers
        return CasadiLike(self._xp(x))

    def zeros_like(self, x: CasadiLike) -> CasadiLike:
        r, c = x.array.shape if len(x.array.shape) == 2 else (x.array.numel(), 1)
        return CasadiLike(self._xp.zeros(r, c))

    def ones_like(self, x: CasadiLike) -> CasadiLike:
        r, c = x.array.shape if len(x.array.shape) == 2 else (x.array.numel(), 1)
        return CasadiLike(self._xp.ones(r, c))

    def tile(self, x: CasadiLike, reps: tuple) -> CasadiLike:
        # No batching in CasADi: return input unchanged.
        return x


class SpatialMath(_SpatialMath):
    """CasADi backend for SpatialMath. Keeps the same high-level API."""

    def __init__(self, spec=None):
        super().__init__(CasadiLikeFactory(spec))

    @staticmethod
    def sin(x: CasadiLike) -> CasadiLike:
        return CasadiLike(cs.sin(x.array))

    @staticmethod
    def cos(x: CasadiLike) -> CasadiLike:
        return CasadiLike(cs.cos(x.array))

    @staticmethod
    def skew(x: Union[CasadiLike, npt.ArrayLike]) -> CasadiLike:
        a = x.array if isinstance(x, CasadiLike) else x
        # Expect 3-vector; if it's a row, transpose; if scalar/empty, raise.
        if isinstance(a, (cs.SX, cs.DM)) and a.is_empty():
            raise ValueError("skew received empty array")
        return CasadiLike(cs.skew(a))

    @staticmethod
    def outer(x: CasadiLike, y: CasadiLike) -> CasadiLike:
        return CasadiLike(cs.np.outer(x.array, y.array))

    @staticmethod
    def vertcat(*x: CasadiLike) -> CasadiLike:
        return CasadiLike(cs.vertcat(*[xi.array for xi in x]))

    @staticmethod
    def horzcat(*x: CasadiLike) -> CasadiLike:
        return CasadiLike(cs.horzcat(*[xi.array for xi in x]))

    @staticmethod
    def stack(x: Sequence[CasadiLike], axis: int = 0) -> CasadiLike:
        arrs = [xi.array for xi in x]
        if axis in {-2, 0}:
            return CasadiLike(cs.vertcat(*arrs))
        if axis in {-1, 1}:
            return CasadiLike(cs.horzcat(*arrs))
        raise NotImplementedError(f"CasADi stack not implemented for axis={axis}")

    @staticmethod
    def concatenate(x: Sequence[CasadiLike], axis: int = 0) -> CasadiLike:
        """
        Concatenate a sequence of CasadiLike objects along a specified axis.

        This function provides flexible concatenation behavior with special handling
        for common use cases in CasADi operations.

        Args:
            x (Sequence[CasadiLike]): Sequence of CasadiLike objects to concatenate
            axis (int, optional): Axis along which to concatenate. Defaults to 0.
                - 0 or -2: Vertical concatenation (stack rows)
                - 1 or -1: Horizontal concatenation (stack columns)

        Returns:
            CasadiLike: The concatenated result

        Raises:
            NotImplementedError: If axis is not in {-2, -1, 0, 1}

        Special Cases:
            - When axis=-1 and exactly 2 column vectors are provided, they are
            vertically stacked to create a longer column vector
            - For horizontal concatenation (axis=1 or -1), if arrays don't have
            matching row dimensions, the function attempts to reshape them:
            * 1D arrays are reshaped to column vectors
            * Row vectors (1xn) are transposed to column vectors
            * Other shapes are kept as-is and stacked vertically

        Note:
            The function uses CasADi's vertcat and horzcat functions internally
            for the actual concatenation operations.
        """
        arrs = [xi.array for xi in x]

        # Friendly special-case: if axis == -1 and we have two column vectors, build a longer column
        if axis == -1 and len(arrs) == 2:
            a, b = arrs
            if (
                len(a.shape) == 2
                and a.shape[1] == 1
                and len(b.shape) == 2
                and b.shape[1] == 1
            ):
                return CasadiLike(cs.vertcat(a, b))

        if axis in {-2, 0}:  # vertical stack
            return CasadiLike(cs.vertcat(*arrs))
        if axis in {-1, 1}:
            if all(arr.shape[0] == arrs[0].shape[0] for arr in arrs):
                return CasadiLike(cs.horzcat(*arrs))
            # Reshape to columns and stack vertically
            cols = []
            for A in arrs:
                if len(A.shape) == 1:
                    cols.append(A.reshape((-1, 1)))
                elif len(A.shape) == 2 and A.shape[1] != 1 and A.shape[0] == 1:
                    cols.append(A.T)
                else:
                    cols.append(A)
            return CasadiLike(cs.vertcat(*cols))
        raise NotImplementedError(f"CasADi concatenate not implemented for axis={axis}")

    @staticmethod
    def swapaxes(x: CasadiLike, axis1: int, axis2: int) -> CasadiLike:
        # Only last-two or (0,1) swaps are meaningful in 2-D CasADi -> transpose.
        if (axis1, axis2) in {(-1, -2), (-2, -1), (0, 1), (1, 0)}:
            return CasadiLike(x.array.T)
        raise NotImplementedError(
            f"CasADi swapaxes not implemented for {axis1=}, {axis2=}"
        )

    def tile(self, x: CasadiLike, reps: tuple) -> CasadiLike:
        # matching ArrayLike API (no-op for CasADi)
        return x

    def transpose(self, x: CasadiLike, dims: tuple) -> CasadiLike:
        # Only 2-D supported; any request means "swap last two"
        return CasadiLike(x.array.T)

    @staticmethod
    def expand_dims(x: CasadiLike, axis: int) -> CasadiLike:
        """Expand dimensions of a CasADi array.

        Args:
            x: Input array (CasadiLike)
            axis: Position where new axis is to be inserted

        Returns:
            CasadiLike: Array with expanded dimensions
        """
        # If axis=-1, we're adding a column dimension to make it (n,1)
        if axis == -1:
            # Reshape to column vector
            return CasadiLike(cs.reshape(x.array, (-1, 1)))
        else:
            # For other axes, just return as is (CasADi is 2D only)
            return x

    @staticmethod
    def inv(x: CasadiLike) -> CasadiLike:
        """Matrix inversion for CasADi.

        Args:
            x: Matrix to invert (CasadiLike)

        Returns:
            CasadiLike: Inverse of x
        """
        return CasadiLike(cs.inv(x.array))

    @staticmethod
    def solve(A: CasadiLike, B: CasadiLike) -> CasadiLike:
        """Solve linear system Ax = B for x using CasADi.

        Args:
            A: Coefficient matrix (CasadiLike)
            B: Right-hand side vector or matrix (CasadiLike)

        Returns:
            CasadiLike: Solution x
        """
        return CasadiLike(cs.solve(A.array, B.array))

    @staticmethod
    def mtimes(A: CasadiLike, B: CasadiLike) -> CasadiLike:
        """Matrix-matrix multiplication for CasADi.

        Args:
            A: First matrix (CasadiLike)
            B: Second matrix (CasadiLike)

        Returns:
            CasadiLike: Result of A @ B
        """
        return CasadiLike(cs.mtimes(A.array, B.array))

    @staticmethod
    def mxv(m: CasadiLike, v: CasadiLike) -> CasadiLike:
        """Matrix-vector multiplication for CasADi.

        Args:
            m: Matrix (CasadiLike)
            v: Vector (CasadiLike)

        Returns:
            CasadiLike: Returns a *column* vector (n,1).
        """
        return CasadiLike(cs.mtimes(m.array, v.array))

    @staticmethod
    def vxs(v: CasadiLike, c: CasadiLike) -> CasadiLike:
        """
        Vector times scalar multiplication for CasADi.
        For multi-DOF joints (v has more than 1 column), performs matrix multiply.

        Args:
            v: Vector or matrix (CasadiLike)
            c: Scalar or vector (CasadiLike)

        Returns:
            CasadiLike: Result of v * c (1-DOF) or v @ c (multi-DOF)
        """
        if v.array.shape[1] > 1:
            return CasadiLike(cs.mtimes(v.array, c.array))
        return CasadiLike(v.array * c.array)
