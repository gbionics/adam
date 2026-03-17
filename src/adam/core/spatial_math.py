import abc
import numpy.typing as npt


class ArrayLike(abc.ABC):
    """Abstract class for a generic Array wrapper. Every method should be implemented for every data type."""

    """This class has to implemented the following operators: """

    @abc.abstractmethod
    def __add__(self, other):
        pass

    @abc.abstractmethod
    def __radd__(self, other):
        pass

    @abc.abstractmethod
    def __sub__(self, other):
        pass

    @abc.abstractmethod
    def __rsub__(self, other):
        pass

    @abc.abstractmethod
    def __mul__(self, other):
        pass

    @abc.abstractmethod
    def __rmul__(self, other):
        pass

    @abc.abstractmethod
    def __matmul__(self, other):
        pass

    @abc.abstractmethod
    def __rmatmul__(self, other):
        pass

    @abc.abstractmethod
    def __neg__(self):
        pass

    @abc.abstractmethod
    def __getitem__(self, item):
        pass

    @abc.abstractmethod
    def __truediv__(self, other):
        pass

    @property
    @abc.abstractmethod
    def T(self):
        """
        Returns: Transpose of the array
        """
        pass

    def __len__(self):
        return len(self.array)

    def __repr__(self):
        return self.array.__repr__()

    def as_list(self):
        for i in range(len(self.array)):
            yield self.array[i]


class ArrayLikeFactory(abc.ABC):
    """Abstract class for a generic Array wrapper. Every method should be implemented for every data type."""

    @abc.abstractmethod
    def zeros(self, *x: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            x (npt.ArrayLike): matrix dimension

        Returns:
            npt.ArrayLike: zero matrix of dimension x
        """
        pass

    @abc.abstractmethod
    def eye(self, x: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            x (npt.ArrayLike): matrix dimension

        Returns:
            npt.ArrayLike: identity matrix of dimension x
        """
        pass

    @abc.abstractmethod
    def asarray(self, x: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            x (npt.ArrayLike): array

        Returns:
            npt.ArrayLike: array
        """
        pass

    @abc.abstractmethod
    def ones_like(self, x: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            x (npt.ArrayLike): array

        Returns:
            npt.ArrayLike: one array with the same shape as x
        """
        pass

    @abc.abstractmethod
    def zeros_like(self, x: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            x (npt.ArrayLike): array

        Returns:
            npt.ArrayLike: one array with the same shape as x
        """
        pass

    @abc.abstractmethod
    def tile(self, x: npt.ArrayLike, reps: tuple) -> npt.ArrayLike:
        """
        Args:
            x (npt.ArrayLike): input array
            reps (tuple): repetition factors for each dimension

        Returns:
            npt.ArrayLike: tiled array
        """
        pass


class SpatialMath:
    """Class implementing the main geometric functions used for computing rigid-body algorithm

    Args:
        ArrayLike: abstract class describing a generic Array wrapper. It needs to be implemented for every data type

    """

    def __init__(self, factory: ArrayLikeFactory):
        self._factory = factory

    @property
    def factory(self) -> ArrayLikeFactory:
        return self._factory

    @abc.abstractmethod
    def vertcat(self, x: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            x (npt.ArrayLike): elements

        Returns:
            npt.ArrayLike: vertical concatenation of elements x
        """
        pass

    @abc.abstractmethod
    def horzcat(self, x: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            x (npt.ArrayLike): elements

        Returns:
            npt.ArrayLike: horizontal concatenation of elements x
        """
        pass

    @abc.abstractmethod
    def concatenate(self, x: npt.ArrayLike, axis: int) -> npt.ArrayLike:
        """
        Args:
            x (npt.ArrayLike): elements
            axis (int): axis along which to concatenate

        Returns:
            npt.ArrayLike: concatenation of elements x along axis
        """
        pass

    @abc.abstractmethod
    def stack(self, x: npt.ArrayLike, axis: int) -> npt.ArrayLike:
        """
        Args:
            x (npt.ArrayLike): elements
            axis (int): axis along which to stack

        Returns:
            npt.ArrayLike: stacked elements x along axis
        """
        pass

    @abc.abstractmethod
    def tile(self, x: npt.ArrayLike, reps: tuple) -> npt.ArrayLike:
        """
        Args:
            x (npt.ArrayLike): input array
            reps (tuple): repetition factors for each dimension

        Returns:
            npt.ArrayLike: tiled array
        """
        pass

    @abc.abstractmethod
    def transpose(self, x: npt.ArrayLike, dims: tuple) -> npt.ArrayLike:
        """
        Args:
            x (npt.ArrayLike): input array
            dims (tuple): permutation of dimensions

        Returns:
            npt.ArrayLike: transposed array
        """
        pass

    @abc.abstractmethod
    def inv(self, x: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            x (npt.ArrayLike): input array

        Returns:
            npt.ArrayLike: inverse of the array
        """
        pass

    @abc.abstractmethod
    def mtimes(self, x: npt.ArrayLike, y: npt.ArrayLike) -> npt.ArrayLike:
        pass

    @abc.abstractmethod
    def sin(self, x: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            x (npt.ArrayLike): angle value

        Returns:
            npt.ArrayLike: sin value of x
        """
        pass

    @abc.abstractmethod
    def cos(self, x: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            x (npt.ArrayLike): angle value

        Returns:
            npt.ArrayLike: cos value of angle x
        """
        pass

    @abc.abstractmethod
    def skew(self, x):
        pass

    def R_from_axis_angle(self, axis, q):
        """
        axis: (...,3) - normalized axis vector, batched if q is batched
        q   : (...,)  - rotation angle, batched or scalar
        returns: (...,3,3) - rotation matrix
        """
        c = self.cos(q)
        s = self.sin(q)

        # Build rotation matrix components
        I = self.factory.eye(q.shape + (3,))
        K = self.skew(axis)  # skew-symmetric matrix
        K_squared = K @ K  # K²
        ones = self.factory.ones_like(c)
        return I + self.sxm(s, K) + self.sxm((ones - c), K_squared)

    def tile(self, x: npt.ArrayLike, reps: tuple) -> npt.ArrayLike:
        """
        Args:
            x (npt.ArrayLike): input array
            reps (tuple): repetition factors for each dimension

        Returns:
            npt.ArrayLike: tiled array
        """
        return self.factory.tile(x, reps)

    def sxm(self, s: npt.ArrayLike, m: npt.ArrayLike) -> npt.ArrayLike:
        """Computes scalar multiplication with a matrix

        Args:
            s (npt.ArrayLike): scalar value
            m (npt.ArrayLike): matrix to be multiplied

        Returns:
            npt.ArrayLike: result of the multiplication
        """
        s = s[..., None, None]
        return s * m

    def Rx(self, q: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            q (npt.ArrayLike): angle value

        Returns:
            npt.ArrayLike: rotation matrix around x axis
        """
        c, s = self.cos(q), self.sin(q)
        one = self.factory.ones_like(c)
        zero = self.factory.zeros_like(c)
        row0 = self.stack([one, zero, zero], axis=-1)
        row1 = self.stack([zero, c, -s], axis=-1)
        row2 = self.stack([zero, s, c], axis=-1)
        return self.stack([row0, row1, row2], axis=-2)

    def Ry(self, q: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            q (npt.ArrayLike): angle value

        Returns:
            npt.ArrayLike: rotation matrix around y axis
        """
        c, s = self.cos(q), self.sin(q)
        one = self.factory.ones_like(c)
        zero = self.factory.zeros_like(c)
        row0 = self.stack([c, zero, s], axis=-1)
        row1 = self.stack([zero, one, zero], axis=-1)
        row2 = self.stack([-s, zero, c], axis=-1)
        return self.stack([row0, row1, row2], axis=-2)

    def Rz(self, q: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            q (npt.ArrayLike): angle value

        Returns:
            npt.ArrayLike: rotation matrix around z axis
        """
        c, s = self.cos(q), self.sin(q)
        one = self.factory.ones_like(c)
        zero = self.factory.zeros_like(c)
        row0 = self.stack([c, -s, zero], axis=-1)
        row1 = self.stack([s, c, zero], axis=-1)
        row2 = self.stack([zero, zero, one], axis=-1)
        return self.stack([row0, row1, row2], axis=-2)

    def H_revolute_joint(
        self,
        xyz: npt.ArrayLike,
        rpy: npt.ArrayLike,
        axis: npt.ArrayLike,
        q: npt.ArrayLike,
    ) -> npt.ArrayLike:
        """
        Args:
            xyz (npt.ArrayLike): joint origin in the urdf
            rpy (npt.ArrayLike): joint orientation in the urdf
            axis (npt.ArrayLike): joint axis in the urdf
            q (npt.ArrayLike): joint angle value

        Returns:
            npt.ArrayLike: Homogeneous transform
        """
        # Check if q is batched to determine if we need to handle batching
        q_is_batched = q.ndim > 0 and q.shape != ()

        if q_is_batched:
            # make all the joint properties batched
            batch_size = q.shape[0]

            # Ensure xyz is batched using standard numpy-style broadcasting
            if xyz.ndim == 1:
                xp = self._xp(xyz.array)
                xyz_batched = xp.tile(xyz.array[None, :], (batch_size, 1))
                xyz = self.factory.asarray(xyz_batched)

            # Ensure rpy is batched
            if rpy.ndim == 1:
                xp = self._xp(rpy.array)
                rpy_batched = xp.tile(rpy.array[None, :], (batch_size, 1))
                rpy = self.factory.asarray(rpy_batched)

            # Ensure axis is batched
            if axis.ndim == 1:
                xp = self._xp(axis.array)
                axis_batched = xp.tile(axis.array[None, :], (batch_size, 1))
                axis = self.factory.asarray(axis_batched)
        R_rpy = self.R_from_RPY(rpy)
        R_axis = self.R_from_axis_angle(axis, q)
        R = R_rpy @ R_axis
        return self.homogeneous(R, xyz)

    def H_spherical_joint(
        self, xyz: npt.ArrayLike, rpy: npt.ArrayLike, q: npt.ArrayLike
    ) -> npt.ArrayLike:
        """
        Args:
            xyz (npt.ArrayLike): joint origin in the urdf
            rpy (npt.ArrayLike): joint orientation in the urdf
            q (npt.ArrayLike): [roll, pitch, yaw] representing joint rotation
        Returns:
            npt.ArrayLike: Homogeneous transform
        """
        if q.shape[-1] != 3:
            raise ValueError(
                f"Spherical joints expect rotation [roll, pitch, yaw] (3 values), received shape {q.shape}"
            )

        if q.ndim > 1:
            batch_shape = q.shape[:-1]
            xp = self._xp(q.array)
            if xyz.ndim == 1:
                xyz = self.factory.asarray(
                    xp.broadcast_to(xyz.array, batch_shape + xyz.array.shape)
                )
            if rpy.ndim == 1:
                rpy = self.factory.asarray(
                    xp.broadcast_to(rpy.array, batch_shape + rpy.array.shape)
                )

        R_rpy = self.R_from_RPY(rpy)
        R_joint = self.R_from_RPY(q)
        R = R_rpy @ R_joint
        return self.homogeneous(R, xyz)

    def homogeneous(self, R, p):
        # Ensure p has the right shape for concatenation
        if p.ndim == R.ndim - 1:
            p = self.factory.asarray(p.array[..., None])  # Add last dimension
        top = self.concatenate([R, p], axis=-1)  # (...,3,4)
        zeros_row = self.factory.zeros_like(R[..., :1, :])  # (...,1,3)
        ones_col = self.factory.ones_like(R[..., :1, :1])  # (...,1,1)
        bottom = self.concatenate([zeros_row, ones_col], axis=-1)  # (...,1,4)
        return self.concatenate([top, bottom], axis=-2)  # (...,4,4)

    def H_prismatic_joint(
        self,
        xyz: npt.ArrayLike,
        rpy: npt.ArrayLike,
        axis: npt.ArrayLike,
        q: npt.ArrayLike,
    ) -> npt.ArrayLike:
        """
        Args:
            xyz (npt.ArrayLike): joint origin in the urdf
            rpy (npt.ArrayLike): joint orientation in the urdf
            axis (npt.ArrayLike): joint axis in the urdf
            q (npt.ArrayLike): joint angle value

        Returns:
            npt.ArrayLike: Homogeneous transform
        """
        R = self.R_from_RPY(rpy)
        p = xyz + q * axis
        return self.homogeneous(R, p)

    def H_from_Pos_RPY(self, xyz: npt.ArrayLike, rpy: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            xyz (npt.ArrayLike): translation vector
            rpy (npt.ArrayLike): rotation as rpy angles

        Returns:
            npt.ArrayLike: Homogeneous transform
        """
        R = self.R_from_RPY(rpy)
        return self.homogeneous(R, xyz)

    def R_from_RPY(self, rpy: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
           rpy (npt.ArrayLike): rotation as rpy angles

        Returns:
            npt.ArrayLike: Rotation matrix
        """
        return self.Rz(rpy[..., 2]) @ self.Ry(rpy[..., 1]) @ self.Rx(rpy[..., 0])

    def X_revolute_joint(
        self,
        xyz: npt.ArrayLike,
        rpy: npt.ArrayLike,
        axis: npt.ArrayLike,
        q: npt.ArrayLike,
    ) -> npt.ArrayLike:
        """
        Args:
            xyz (npt.ArrayLike): joint origin in the urdf
            rpy (npt.ArrayLike): joint orientation in the urdf
            axis (npt.ArrayLike): joint axis in the urdf
            q (npt.ArrayLike): joint angle value

        Returns:
            npt.ArrayLike: Spatial transform of a revolute joint given its rotation angle
        """
        # TODO: give Featherstone reference
        T = self.H_revolute_joint(xyz, rpy, axis, q)
        R = self.swapaxes(T[..., :3, :3], -1, -2)
        p = self.mxv(-R, T[..., :3, 3])
        return self.spatial_transform(R, p)

    def X_spherical_joint(
        self,
        xyz: npt.ArrayLike,
        rpy: npt.ArrayLike,
        q: npt.ArrayLike,
    ) -> npt.ArrayLike:
        """
        Args:
            xyz (npt.ArrayLike): joint origin in the urdf
            rpy (npt.ArrayLike): joint orientation in the urdf
            q (npt.ArrayLike): [roll, pitch, yaw] representing joint rotation

        Returns:
            npt.ArrayLike: Spatial transform of a spherical joint given its rotation angles
        """
        T = self.H_spherical_joint(xyz, rpy, q)
        R = self.swapaxes(T[..., :3, :3], -1, -2)
        p = self.mxv(-R, T[..., :3, 3])
        return self.spatial_transform(R, p)

    def X_prismatic_joint(
        self,
        xyz: npt.ArrayLike,
        rpy: npt.ArrayLike,
        axis: npt.ArrayLike,
        q: npt.ArrayLike,
    ) -> npt.ArrayLike:
        """
        Args:
            xyz (npt.ArrayLike): joint origin in the urdf
            rpy (npt.ArrayLike): joint orientation in the urdf
            axis (npt.ArrayLike): joint axis in the urdf
            q (npt.ArrayLike): joint angle value

        Returns:
            npt.ArrayLike: Spatial transform of a prismatic joint given its increment
        """
        T = self.H_prismatic_joint(xyz, rpy, axis, q)
        R = T[:3, :3].T
        p = -T[:3, :3].T @ T[:3, 3]
        return self.spatial_transform(R, p)

    def X_fixed_joint(self, xyz: npt.ArrayLike, rpy: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            xyz (npt.ArrayLike): joint origin in the urdf
            rpy (npt.ArrayLike): joint orientation in the urdf

        Returns:
            npt.ArrayLike: Spatial transform of a fixed joint
        """
        T = self.H_from_Pos_RPY(xyz, rpy)
        R = T[:3, :3].T
        p = -T[:3, :3].T @ T[:3, 3]
        return self.spatial_transform(R, p)

    def _X_from_H(self, T):
        R = self.swapaxes(T[..., :3, :3], -1, -2)
        p = -(R @ T[..., :3, 3:4])[..., :, 0]
        return self.spatial_transform(R, p)

    def spatial_transform(self, R: npt.ArrayLike, p: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            R (npt.ArrayLike): Rotation matrix
            p (npt.ArrayLike): translation vector

        Returns:
            npt.ArrayLike: spatial transform
        """

        Sp = self.skew(p)  # (...,3,3)
        zeros = self.factory.zeros_like(R)
        top = self.concatenate([R, Sp @ R], axis=-1)  # (...,3,6)
        bottom = self.concatenate([zeros, R], axis=-1)  # (...,3,6)
        return self.concatenate([top, bottom], axis=-2)  # (...,6,6)

    def spatial_inertia(
        self,
        inertia_matrix: npt.ArrayLike,
        mass: npt.ArrayLike,
        c: npt.ArrayLike,
        rpy: npt.ArrayLike,
    ) -> npt.ArrayLike:
        """
        Args:
            inertia_matrix (npt.ArrayLike): inertia values from urdf
            mass (npt.ArrayLike): mass value from urdf
            c (npt.ArrayLike): origin of the link from urdf
            rpy (npt.ArrayLike): orientation of the link from the urdf

        Returns:
            npt.ArrayLike: the 6x6 inertia matrix expressed at the origin of the link (with rotation)
        """
        # Compute components
        Sc = self.skew(c)
        R = self.R_from_RPY(rpy)
        mass_I3 = self.sxm(mass, self.factory.eye(3))
        mass_Sc = self.sxm(mass, Sc)
        mass_Sc_T = self.swapaxes(mass_Sc, -1, -2)

        rotated_inertia = R @ inertia_matrix @ self.swapaxes(R, -1, -2)
        Sc_squared = Sc @ self.swapaxes(Sc, -1, -2)
        bottom_right = rotated_inertia + self.sxm(mass, Sc_squared)

        # Correct block placement:
        top = self.concatenate([mass_I3, mass_Sc_T], axis=-1)  # (...,3,6)
        bottom = self.concatenate([mass_Sc, bottom_right], axis=-1)  # (...,3,6)
        return self.concatenate([top, bottom], axis=-2)  # (...,6,6)

    def spatial_inertia_with_parameters(self, inertia_matrix, mass, c, rpy):
        """
        Args:
            I (npt.ArrayLike): inertia values parametric
            mass (npt.ArrayLike): mass value parametric
            c (npt.ArrayLike): origin of the link parametric
            rpy (npt.ArrayLike): orientation of the link from urdf

        Returns:
            npt.ArrayLike: the 6x6 inertia matrix parametric expressed at the origin of the link (with rotation)
        """
        Sc = self.skew(c)
        R = self.R_from_RPY(rpy)

        mass_I3 = self.sxm(mass, self.factory.eye(3))
        mass_Sc = self.sxm(mass, Sc)
        mass_Sc_T = self.swapaxes(mass_Sc, -1, -2)

        rotated_inertia = R @ inertia_matrix @ self.swapaxes(R, -1, -2)
        Sc_squared = Sc @ self.swapaxes(Sc, -1, -2)
        bottom_right = rotated_inertia + self.sxm(mass, Sc_squared)

        # Correct block placement:
        top = self.concatenate([mass_I3, mass_Sc_T], axis=-1)  # (...,3,6)
        bottom = self.concatenate([mass_Sc, bottom_right], axis=-1)  # (...,3,6)
        return self.concatenate([top, bottom], axis=-2)  # (...,6,6)

    def spatial_skew(self, v: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            v (npt.ArrayLike): 6D vector

        Returns:
            npt.ArrayLike: spatial skew matrix
        """
        omega = v[..., 3:]  # Angular part
        vel = v[..., :3]  # Linear part

        skew_omega = self.skew(omega)
        skew_vel = self.skew(vel)
        zeros = self.factory.zeros_like(skew_omega)

        top = self.concatenate([skew_omega, skew_vel], axis=-1)  # (...,3,6)
        bottom = self.concatenate([zeros, skew_omega], axis=-1)  # (...,3,6)
        return self.concatenate([top, bottom], axis=-2)  # (...,6,6)

    def spatial_skew_star(self, v: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            v (npt.ArrayLike): 6D vector

        Returns:
            npt.ArrayLike: negative spatial skew matrix traspose
        """
        # return -self.spatial_skew(v).T
        return -self.swapaxes(self.spatial_skew(v), -1, -2)

    def adjoint(self, H: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            H (npt.ArrayLike): Homogeneous transform
        Returns:
            npt.ArrayLike: adjoint matrix
        """
        R = H[..., :3, :3]
        p = H[..., :3, 3]
        return self.spatial_transform(R, p)

    def adjoint_derivative(self, H: npt.ArrayLike, v: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            H (npt.ArrayLike): Homogeneous transform
            v (npt.ArrayLike): 6D twist
        Returns:
            npt.ArrayLike: adjoint matrix derivative
        """

        R = H[..., :3, :3]
        p = H[..., :3, 3]
        v_linear = v[..., :3]
        v_angular = v[..., 3:]
        Rdot = self.skew(v_angular) @ R
        if v_angular.shape[-1] == 3:
            # promote to column for consistent matmul semantics
            omega_col = v_angular[..., None]
            v_linear = v_linear[..., None]
        else:
            omega_col = v_angular
            v_linear = v_linear

        pdot = v_linear - self.skew(p) @ omega_col

        Z = self.factory.zeros_like(R)
        S = self.skew(pdot) @ R + self.skew(p) @ Rdot
        top = self.concatenate([Rdot, S], axis=-1)  # (...,3,6)
        bottom = self.concatenate([Z, Rdot], axis=-1)  # (...,3,6)
        return self.concatenate([top, bottom], axis=-2)  # (...,6,6)

    def mxv(self, m: npt.ArrayLike, v: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            m (npt.ArrayLike): Matrix
            v (npt.ArrayLike): Vector
        Returns:
            npt.ArrayLike: Result of matrix-vector multiplication
        """
        res = m @ v[..., None]
        return res[..., 0]  # Remove the extra dimension

    def vxs(self, v: npt.ArrayLike, s: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            v (npt.ArrayLike): Vector
            s (npt.ArrayLike): Scalar
        Returns:
            npt.ArrayLike: Result of vector cross product with scalar multiplication
        """
        if v.shape[-1] == 1:
            v = v[..., 0]
        s = s[..., None]  # Add extra dimension
        return v * s

    def adjoint_inverse(self, H: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            H (npt.ArrayLike): Homogeneous transform
        Returns:
            npt.ArrayLike: adjoint matrix
        """
        R = H[..., :3, :3]
        p = H[..., :3, 3:4]
        RT = self.swapaxes(R, -1, -2)
        return self.spatial_transform(RT, -(RT @ p)[..., :, 0])

    def adjoint_inverse_derivative(
        self, H: npt.ArrayLike, v: npt.ArrayLike
    ) -> npt.ArrayLike:
        """
        Args:
            H (npt.ArrayLike): Homogeneous transform
            v (npt.ArrayLike): 6D twist
        Returns:
            npt.ArrayLike: adjoint matrix derivative
        """
        R = H[..., :3, :3]
        p = H[..., :3, 3]
        R_dot = self.skew(v[..., 3:]) @ R
        p_dot = v[..., :3] - self.skew(p) @ v[..., 3:]
        R_T = self.swapaxes(R, -1, -2)
        R_dot_T = self.swapaxes(R_dot, -1, -2)
        Z = self.factory.zeros_like(R)
        TR = -R_dot_T @ self.skew(p) - R_T @ self.skew(p_dot)
        top = self.concatenate([R_dot_T, TR], axis=-1)  # (...,3,6)
        bottom = self.concatenate([Z, R_dot_T], axis=-1)  # (...,3,6)
        return self.concatenate([top, bottom], axis=-2)  # (...,6,6)

    def adjoint_mixed(self, H: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            H (npt.ArrayLike): Homogeneous transform
        Returns:
            npt.ArrayLike: adjoint matrix
        """
        R = H[..., :3, :3]
        Z = self.factory.zeros_like(R)
        return self.concatenate(
            [
                self.concatenate([R, Z], axis=-1),
                self.concatenate([Z, R], axis=-1),
            ],
            axis=-2,
        )

    def adjoint_mixed_inverse(self, H: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            H (npt.ArrayLike): Homogeneous transform
        Returns:
            npt.ArrayLike: adjoint matrix
        """
        RT = self.swapaxes(H[..., :3, :3], -1, -2)
        Z = self.zeros_like(RT)
        return self.concatenate(
            [self.concatenate([RT, Z], axis=-1), self.concatenate([Z, RT], axis=-1)],
            axis=-2,
        )

    def adjoint_mixed_derivative(
        self, H: npt.ArrayLike, v: npt.ArrayLike
    ) -> npt.ArrayLike:
        """
        Args:
            H (npt.ArrayLike): Homogeneous transform
            v (npt.ArrayLike): 6D twist
        Returns:
            npt.ArrayLike: adjoint matrix derivative
        """
        R = H[..., :3, :3]
        omega = v[..., 3:]
        R_dot = self.skew(omega) @ R
        Z = self.factory.zeros_like(R_dot)
        top = self.concatenate([R_dot, Z], axis=-1)  # (...,3,6)
        bottom = self.concatenate([Z, R_dot], axis=-1)  # (...,3,6)
        return self.concatenate([top, bottom], axis=-2)  # (...,6,6)

    def adjoint_mixed_inverse_derivative(
        self, H: npt.ArrayLike, v: npt.ArrayLike
    ) -> npt.ArrayLike:
        """
        Args:
            H (npt.ArrayLike): Homogeneous transform
            v (npt.ArrayLike): 6D twist
        Returns:
            npt.ArrayLike: adjoint matrix derivative
        """
        R = H[:3, :3]
        R_dot = self.skew(v[3:]) @ R
        X = self.factory.zeros(6, 6)
        X[:3, :3] = R_dot.T
        X[3:6, 3:6] = R_dot.T
        return X

    def homogeneous_inverse(self, H: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            H (npt.ArrayLike): Homogeneous transform
        Returns:
            npt.ArrayLike: inverse of the homogeneous transform
        """
        R = H[..., :3, :3]  # (...,3,3)
        p = H[..., :3, 3:4]  # (...,3,1)
        R_T = self.swapaxes(R, -1, -2)  # (...,3,3)
        Rp = -(R_T @ p)  # (...,3,1)

        top = self.concatenate([R_T, Rp], axis=-1)  # (...,3,4)

        last_row = self.factory.zeros(H.shape[:-2] + (1, 4))
        last_row = last_row + self.factory.asarray([0, 0, 0, 1])

        return self.concatenate([top, last_row], axis=-2)  # (...,4,4)

    def zeros(self, *x: int) -> npt.ArrayLike:
        """
        Args:
            x (int): dimension
        Returns:
            npt.ArrayLike: zero matrix of dimension x
        """
        return self.factory.zeros(*x)

    def eye(self, x: int) -> npt.ArrayLike:
        """
        Args:
            x (int): dimension
        Returns:
            npt.ArrayLike: identity matrix of dimension x
        """
        return self.factory.eye(x)

    def asarray(self, x: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            x (npt.ArrayLike): array
        Returns:
            npt.ArrayLike: array
        """
        return self.factory.asarray(x)

    def zeros_like(self, x: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            x (npt.ArrayLike): array
        Returns:
            npt.ArrayLike: zero array with the same shape as x
        """
        return self.factory.zeros_like(x)

    def ones_like(self, x: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            x (npt.ArrayLike): array
        Returns:
            npt.ArrayLike: one array with the same shape as x
        """
        return self.factory.ones_like(x)
