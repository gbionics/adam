# Copyright (C) Istituto Italiano di Tecnologia (IIT). All rights reserved.

from dataclasses import dataclass

import numpy.typing as npt

from adam.core.constants import Representations
from adam.core.spatial_math import ArrayLike, SpatialMath
from adam.model import Model, Node, Joint


@dataclass(frozen=True, slots=True)
class ChainTraversal:
    """Homogeneous transforms accumulated along a root->frame chain."""

    joints: tuple[Joint, ...]
    root_to_joint: tuple[ArrayLike, ...]
    root_to_target: ArrayLike


class RBDAlgorithms:
    """This is a small class that implements Rigid body algorithms retrieving robot quantities, for Floating Base systems - as humanoid robots."""

    def __init__(self, model: Model, math: SpatialMath) -> None:
        """
        Args:
            model (Model): the adam.model representing the robot
            math (SpatialMath): the spatial math.
        """

        self.model = model
        self.NDoF = model.NDoF
        self.root_link = self.model.tree.root
        self.math = math
        self.frame_velocity_representation = (
            Representations.MIXED_REPRESENTATION
        )  # default
        # Cache root quantities that are reused at every call
        self._root_spatial_transform = self.math.spatial_transform(
            self.math.factory.eye(3),
            self.math.factory.zeros((3, 1)),
        )
        self._root_motion_subspace = self.math.factory.eye(6)
        self._prepare_tree_cache()

    def set_frame_velocity_representation(self, representation: Representations):
        """Sets the frame velocity representation

        Args:
            representation (str): The representation of the frame velocity
        """
        self.frame_velocity_representation = representation

    def crba(self, base_transform: npt.ArrayLike, joint_positions: npt.ArrayLike):
        """
        Batched Composite Rigid Body Algorithm (CRBA) + Orin's Centroidal Momentum Matrix.

        Args:
            base_transform (npt.ArrayLike): The homogenous transform from base to world frame
            joint_positions (npt.ArrayLike): The joints position

        Returns:
            M, Jcm (npt.ArrayLike, npt.ArrayLike): The mass matrix and the centroidal momentum matrix
        """
        base_transform, joint_positions = self._convert_to_arraylike(
            base_transform, joint_positions
        )

        math = self.math
        n = self.NDoF
        node_count = self._node_count
        node_indices = self._node_indices
        rev_indices = self._rev_node_indices
        parent_indices = self._parent_indices
        joint_indices = self._joint_indices_per_node
        motion_subspaces = self._motion_subspaces
        inertias = self._spatial_inertias
        joints = self._joints_per_node
        root_idx = self._root_index
        joint_to_node = self._joint_index_to_node

        if len(joint_positions.shape) >= 2:
            batch_shape = joint_positions.shape[:-1]
        elif base_transform.ndim > 2:
            batch_shape = base_transform.shape[:-2]
        else:
            batch_shape = ()

        if joint_positions.shape[-1] > 0:
            zero_q = math.zeros_like(joint_positions[..., 0])
        else:
            zero_q = math.zeros_like(base_transform[..., 0, 0])

        def tile_batch(arr):
            if not batch_shape:
                return arr
            reps = batch_shape + (1,) * len(arr.shape)
            return math.tile(arr, reps)

        Xup = [None] * node_count
        Phi = [None] * node_count
        Ic = [None] * node_count

        for idx in node_indices:
            Ic[idx] = tile_batch(inertias[idx])
            if idx == root_idx:
                Xup[idx] = tile_batch(self._root_spatial_transform)
                Phi[idx] = tile_batch(self._root_motion_subspace)
            else:
                joint = joints[idx]
                joint_idx = joint_indices[idx]
                q_i = (
                    joint_positions[..., joint_idx] if joint_idx is not None else zero_q
                )
                X_current = (
                    joint.spatial_transform(q=q_i)
                    if joint is not None
                    else self._root_spatial_transform
                )
                Xup[idx] = X_current
                Phi[idx] = tile_batch(motion_subspaces[idx])

        Xup_T = [math.swapaxes(Xup[i], -2, -1) for i in range(node_count)]
        Ic_comp = Ic[:]

        for idx in rev_indices:
            parent = parent_indices[idx]
            if parent >= 0:
                Ic_comp[parent] = Ic_comp[parent] + math.mtimes(
                    math.mtimes(Xup_T[idx], Ic_comp[idx]),
                    Xup[idx],
                )

        def block_info(node_idx: int) -> list[tuple[int, int]] | None:
            """Return list of (block_index, local_dof_offset) for a node, or None."""
            if node_idx == root_idx:
                return [(0, 0)]
            joint_idx = joint_indices[node_idx]
            if joint_idx is None:
                return None
            if isinstance(joint_idx, tuple):
                return [(1 + j, k) for k, j in enumerate(joint_idx)]
            return [(1 + int(joint_idx), 0)]

        sizes = [6] + [1] * n
        blocks: list[list[npt.ArrayLike | None]] = [
            [None for _ in range(n + 1)] for _ in range(n + 1)
        ]

        for idx in rev_indices:
            Phi_i = Phi[idx]
            Phi_T = math.swapaxes(Phi_i, -2, -1)
            F = math.mtimes(Ic_comp[idx], Phi_i)
            ri_info = block_info(idx)

            if idx == root_idx:
                blocks[0][0] = math.mtimes(Phi_T, F)
                continue

            if ri_info is not None:
                diag = math.mtimes(Phi_T, F)
                for ri, oi in ri_info:
                    for rj, oj in ri_info:
                        blocks[ri][rj] = diag[
                            ..., oi : oi + sizes[ri], oj : oj + sizes[rj]
                        ]

            current = idx
            F_path = F
            while current != root_idx:
                parent = parent_indices[current]
                F_path = math.mtimes(Xup_T[current], F_path)
                rj_info = block_info(parent)
                if rj_info is None:
                    current = parent
                    continue
                Bij_full = math.mtimes(math.swapaxes(F_path, -2, -1), Phi[parent])
                if ri_info is None:
                    current = parent
                    continue
                for ri, oi in ri_info:
                    for rj, oj in rj_info:
                        sub = Bij_full[..., oi : oi + sizes[ri], oj : oj + sizes[rj]]
                        blocks[ri][rj] = sub
                        blocks[rj][ri] = math.swapaxes(sub, -2, -1)
                current = parent

        row_tensors = []
        for r in range(n + 1):
            row_blocks = []
            for c in range(n + 1):
                block = blocks[r][c]
                if block is None:
                    block = math.factory.zeros(batch_shape + (sizes[r], sizes[c]))
                row_blocks.append(block)
            row_tensors.append(math.concatenate(row_blocks, axis=-1))
        M = math.concatenate(row_tensors, axis=-2)

        A = math.swapaxes(M[..., :3, 3:6], -2, -1) / M[..., 0, 0][..., None, None]
        I3 = math.factory.eye(batch_shape + (3,))
        Z3 = math.factory.zeros(batch_shape + (3, 3))
        top = math.concatenate([I3, A], axis=-1)
        bot = math.concatenate([Z3, I3], axis=-1)
        O_X_G = math.concatenate([top, bot], axis=-2)

        X_G = [None] * node_count
        for idx in node_indices:
            if idx == root_idx:
                X_G[idx] = O_X_G
            else:
                parent = parent_indices[idx]
                X_G[idx] = math.mtimes(Xup[idx], X_G[parent])

        J_base = math.mtimes(
            math.swapaxes(X_G[root_idx], -2, -1),
            math.mtimes(Ic_comp[root_idx], Phi[root_idx]),
        )

        zero_col = math.factory.zeros(batch_shape + (6, 1))
        joint_local_dof = self._joint_index_to_local_dof
        joint_cols = []
        for jidx in range(n):
            node_idx = joint_to_node.get(jidx)
            if node_idx is not None:
                full_col = math.mtimes(
                    math.swapaxes(X_G[node_idx], -2, -1),
                    math.mtimes(Ic_comp[node_idx], Phi[node_idx]),
                )
                local = joint_local_dof.get(jidx, 0)
                col = full_col[..., :, local : local + 1]
            else:
                col = zero_col
            joint_cols.append(col)

        Jcm = math.concatenate([J_base] + joint_cols, axis=-1)

        if (
            self.frame_velocity_representation
            == Representations.BODY_FIXED_REPRESENTATION
        ):
            return M, Jcm

        if self.frame_velocity_representation == Representations.MIXED_REPRESENTATION:
            Xm = math.adjoint_mixed_inverse(base_transform)
            In = math.factory.eye(batch_shape + (n,))
            Z6n = math.factory.zeros(batch_shape + (6, n))
            Zn6 = math.factory.zeros(batch_shape + (n, 6))

            top = math.concatenate([Xm, Z6n], axis=-1)
            bot = math.concatenate([Zn6, In], axis=-1)
            X_to_mixed = math.concatenate([top, bot], axis=-2)

            M_mixed = math.mtimes(
                math.swapaxes(X_to_mixed, -2, -1), math.mtimes(M, X_to_mixed)
            )
            Jcm_mixed = math.mtimes(
                math.swapaxes(Xm, -2, -1), math.mtimes(Jcm, X_to_mixed)
            )
            return M_mixed, Jcm_mixed

        if (
            self.frame_velocity_representation
            == Representations.INERTIAL_FIXED_REPRESENTATION
        ):
            Xi = math.adjoint_inverse(base_transform)
            In = math.factory.eye(batch_shape + (n,))
            Z6n = math.factory.zeros(batch_shape + (6, n))
            Zn6 = math.factory.zeros(batch_shape + (n, 6))

            top = math.concatenate([Xi, Z6n], axis=-1)
            bot = math.concatenate([Zn6, In], axis=-1)
            X_to_inertial = math.concatenate([top, bot], axis=-2)

            M_inertial = math.mtimes(
                math.swapaxes(X_to_inertial, -2, -1), math.mtimes(M, X_to_inertial)
            )
            Xm = math.adjoint_mixed_inverse(base_transform)  # rotation-only, for cmm
            wrench_transform = math.swapaxes(Xm, -2, -1)
            Jcm_inertial = math.mtimes(
                wrench_transform, math.mtimes(Jcm, X_to_inertial)
            )
            return M_inertial, Jcm_inertial

        raise ValueError(
            f"Unknown frame velocity representation: {self.frame_velocity_representation}"
        )

    def forward_kinematics(
        self, frame, base_transform: npt.ArrayLike, joint_positions: npt.ArrayLike
    ) -> npt.ArrayLike:
        """Computes the forward kinematics relative to the specified `frame`.

        Args:
            frame (str): The frame to which the fk will be computed
            base_transform (npt.ArrayLike): The homogenous transform from base to world frame
            joint_positions (npt.ArrayLike): The joints position

        Returns:
            I_H_L (npt.ArrayLike): The fk represented as Homogenous transformation matrix
        """
        base_transform, joint_positions = self._convert_to_arraylike(
            base_transform, joint_positions
        )
        return self._compute_chain_homogeneous_traversal(
            frame=frame,
            base_transform=base_transform,
            joint_positions=joint_positions,
        ).root_to_target

    def joints_jacobian(
        self, frame: str, joint_positions: npt.ArrayLike
    ) -> npt.ArrayLike:
        """Returns the Jacobian relative to the specified `frame`.

        Args:
            frame (str): The frame to which the jacobian will be computed
            base_transform (npt.ArrayLike): The homogenous transform from base to world frame
            joint_positions (npt.ArrayLike): The joints position

        Returns:
            J (npt.ArrayLike): The Joints Jacobian relative to the frame
        """
        joint_positions = self._convert_to_arraylike(joint_positions)

        batch_size = (
            joint_positions.shape[:-1] if len(joint_positions.shape) >= 2 else ()
        )

        eye = self.math.factory.eye(batch_size + (4,))
        traversal = self._compute_chain_homogeneous_traversal(
            frame=frame,
            base_transform=eye,
            joint_positions=joint_positions,
        )
        return self._joints_jacobian_from_traversal(
            traversal=traversal,
            joint_positions=joint_positions,
        )

    def jacobian(
        self, frame: str, base_transform: npt.ArrayLike, joint_positions: npt.ArrayLike
    ) -> npt.ArrayLike:
        """Returns the Jacobian for `frame`.

        Args:
            frame (str): The frame to which the jacobian will be computed
            base_transform (npt.ArrayLike): The homogenous transform from base to world frame
            joint_positions (npt.ArrayLike): The joints position

        Returns:
            npt.ArrayLike: The Jacobian for the specified frame
        """

        base_transform, joint_positions = self._convert_to_arraylike(
            base_transform, joint_positions
        )

        batch_size = (
            joint_positions.shape[:-1] if len(joint_positions.shape) >= 2 else ()
        )
        eye = self.math.factory.eye(batch_size + (4,))
        traversal = self._compute_chain_homogeneous_traversal(
            frame=frame,
            base_transform=eye,
            joint_positions=joint_positions,
        )
        J = self._joints_jacobian_from_traversal(
            traversal=traversal,
            joint_positions=joint_positions,
        )
        B_H_L = traversal.root_to_target
        L_X_B = self.math.adjoint_inverse(B_H_L)
        J_tot = self.math.concatenate([L_X_B, J], axis=-1)
        w_H_B = base_transform
        if (
            self.frame_velocity_representation
            == Representations.BODY_FIXED_REPRESENTATION
        ):
            return J_tot
        if self.frame_velocity_representation == Representations.MIXED_REPRESENTATION:
            w_H_L = w_H_B @ B_H_L
            LI_X_L = self.math.adjoint_mixed(w_H_L)

            top_left = self.math.adjoint_mixed_inverse(base_transform)
            top_right = self.math.factory.zeros(
                joint_positions.shape[:-1] + (6, self.NDoF)
            )
            bottom_left = self.math.factory.zeros(
                joint_positions.shape[:-1] + (self.NDoF, 6)
            )
            bottom_right = self.math.factory.eye(
                joint_positions.shape[:-1] + (self.NDoF,)
            )
            top = self.math.concatenate([top_left, top_right], axis=-1)
            bottom = self.math.concatenate([bottom_left, bottom_right], axis=-1)
            X = self.math.concatenate([top, bottom], axis=-2)

            J_tot = LI_X_L @ J_tot @ X
            return J_tot

        if (
            self.frame_velocity_representation
            == Representations.INERTIAL_FIXED_REPRESENTATION
        ):
            w_H_L = w_H_B @ B_H_L
            I_X_L = self.math.adjoint(w_H_L)

            top_left = self.math.adjoint_inverse(base_transform)
            top_right = self.math.factory.zeros(
                joint_positions.shape[:-1] + (6, self.NDoF)
            )
            bottom_left = self.math.factory.zeros(
                joint_positions.shape[:-1] + (self.NDoF, 6)
            )
            bottom_right = self.math.factory.eye(
                joint_positions.shape[:-1] + (self.NDoF,)
            )
            top = self.math.concatenate([top_left, top_right], axis=-1)
            bottom = self.math.concatenate([bottom_left, bottom_right], axis=-1)
            X = self.math.concatenate([top, bottom], axis=-2)

            return I_X_L @ J_tot @ X

        raise NotImplementedError(
            "Only BODY_FIXED_REPRESENTATION, MIXED_REPRESENTATION and INERTIAL_FIXED_REPRESENTATION are implemented"
        )

    def relative_jacobian(
        self, frame: str, joint_positions: npt.ArrayLike
    ) -> npt.ArrayLike:
        """Returns the Jacobian between the root link and a specified `frame`.

        Args:
            frame (str): The tip of the chain
            joint_positions (npt.ArrayLike): The joints position

        Returns:
            J (npt.ArrayLike): The 6 x NDoF Jacobian between the root and the `frame`
        """
        joint_positions = self._convert_to_arraylike(joint_positions)
        batch_size = (
            joint_positions.shape[:-1] if len(joint_positions.shape) >= 2 else ()
        )
        eye = self.math.factory.eye(batch_size + (4,))
        traversal = self._compute_chain_homogeneous_traversal(
            frame=frame,
            base_transform=eye,
            joint_positions=joint_positions,
        )
        J = self._joints_jacobian_from_traversal(
            traversal=traversal,
            joint_positions=joint_positions,
        )
        if (
            self.frame_velocity_representation
            == Representations.BODY_FIXED_REPRESENTATION
        ):
            return J
        elif self.frame_velocity_representation == Representations.MIXED_REPRESENTATION:
            B_H_L = traversal.root_to_target
            LI_X_L = self.math.adjoint_mixed(B_H_L)
            return LI_X_L @ J

        elif (
            self.frame_velocity_representation
            == Representations.INERTIAL_FIXED_REPRESENTATION
        ):
            B_H_L = traversal.root_to_target
            I_X_L = self.math.adjoint(B_H_L)
            return I_X_L @ J

        raise NotImplementedError(
            "Only BODY_FIXED_REPRESENTATION, MIXED_REPRESENTATION and INERTIAL_FIXED_REPRESENTATION are implemented"
        )

    def jacobian_dot(
        self,
        frame: str,
        base_transform: npt.ArrayLike,
        joint_positions: npt.ArrayLike,
        base_velocity: npt.ArrayLike,
        joint_velocities: npt.ArrayLike,
    ) -> npt.ArrayLike:
        """Returns the Jacobian time derivative for `frame`.

        Args:
            frame (str): The frame to which the jacobian will be computed
            base_transform (npt.ArrayLike): The homogenous transform from base to world frame
            joint_positions (npt.ArrayLike): The joints position
            base_velocity (npt.ArrayLike): The spatial velocity of the base
            joint_velocities (npt.ArrayLike): The joints velocities

        Returns:
            J_dot (npt.ArrayLike): The Jacobian derivative relative to the frame

        """

        base_transform, joint_positions, base_velocity, joint_velocities = (
            self._convert_to_arraylike(
                base_transform, joint_positions, base_velocity, joint_velocities
            )
        )

        batch_size = base_transform.shape[:-2] if len(base_transform.shape) > 2 else ()

        I4 = self.math.factory.eye(batch_size + (4,))
        traversal = self._compute_chain_homogeneous_traversal(
            frame=frame,
            base_transform=I4,
            joint_positions=joint_positions,
        )
        B_H_L = traversal.root_to_target
        L_H_B = self.math.homogeneous_inverse(B_H_L)

        if self.frame_velocity_representation == Representations.MIXED_REPRESENTATION:
            # frame C = B[I] for mixed representation
            B_v_C = self.math.mxv(
                self.math.adjoint_mixed_inverse(base_transform), base_velocity
            )
        elif (
            self.frame_velocity_representation
            == Representations.INERTIAL_FIXED_REPRESENTATION
        ):
            # frame C = I for inertial-fixed representation
            B_v_C = self.math.mxv(
                self.math.adjoint_inverse(base_transform), base_velocity
            )
        elif (
            self.frame_velocity_representation
            == Representations.BODY_FIXED_REPRESENTATION
        ):
            # frame C = B for body-fixed representation
            B_v_C = base_velocity
        else:
            raise NotImplementedError(
                "Only BODY_FIXED_REPRESENTATION, MIXED_REPRESENTATION, and INERTIAL_FIXED_REPRESENTATION are implemented"
            )

        v = self.math.mxv(self.math.adjoint(L_H_B), B_v_C)
        a = self.math.mxv(self.math.adjoint_derivative(L_H_B, v), B_v_C)

        J_base_full = self.math.adjoint_inverse(B_H_L)
        J_base_cols = [J_base_full[..., :, i : i + 1] for i in range(6)]
        J_dot_base_full = self.math.adjoint_derivative(L_H_B, v)
        J_dot_base_cols = [J_dot_base_full[..., :, i : i + 1] for i in range(6)]

        cols: list = [None] * self.NDoF
        cols_dot: list = [None] * self.NDoF

        for joint, B_H_j in zip(traversal.joints, traversal.root_to_joint):
            if joint.idx is None:
                continue

            q_dot = joint_velocities[..., joint.idx]
            L_H_j = L_H_B @ B_H_j
            S = joint.motion_subspace()
            J_j = self.math.adjoint(L_H_j) @ S

            v = v + self.math.vxs(J_j, q_dot)
            J_dot_j = self.math.adjoint_derivative(L_H_j, v) @ S
            a = a + self.math.vxs(J_dot_j, q_dot)

            if joint.idx is not None:
                self._scatter_cols(cols, joint.idx, J_j)
                self._scatter_cols(cols_dot, joint.idx, J_dot_j)

        zero_col = self.math.factory.zeros(batch_size + (6, 1))
        cols = [zero_col if c is None else c for c in cols]
        cols_dot = [zero_col if c is None else c for c in cols_dot]

        J = self.math.concatenate([*J_base_cols, *cols], axis=-1)
        J_dot = self.math.concatenate([*J_dot_base_cols, *cols_dot], axis=-1)

        if (
            self.frame_velocity_representation
            == Representations.BODY_FIXED_REPRESENTATION
        ):
            return J_dot

        elif self.frame_velocity_representation == Representations.MIXED_REPRESENTATION:
            adj = self.math.adjoint_mixed
            adj_derivative = self.math.adjoint_mixed_derivative
            adj_inverse = self.math.adjoint_mixed_inverse
        elif (
            self.frame_velocity_representation
            == Representations.INERTIAL_FIXED_REPRESENTATION
        ):
            adj = self.math.adjoint
            adj_derivative = self.math.adjoint_derivative
            adj_inverse = self.math.adjoint_inverse
        else:
            raise NotImplementedError(
                "Only BODY_FIXED_REPRESENTATION, MIXED_REPRESENTATION and INERTIAL_FIXED_REPRESENTATION are implemented"
            )
        I_H_L = base_transform @ B_H_L
        I_X_L = adj(I_H_L)
        I_v_L = self.math.mxv(I_X_L, v)
        I_X_L_dot = adj_derivative(I_H_L, I_v_L)

        adj_inv = adj_inverse(base_transform)

        Z_6xN = self.math.factory.zeros(batch_size + (6, self.NDoF))
        Z_Nx6 = self.math.factory.zeros(batch_size + (self.NDoF, 6))
        I_N = self.math.factory.eye(batch_size + (self.NDoF,))

        top = self.math.concatenate([adj_inv, Z_6xN], axis=-1)
        bottom = self.math.concatenate([Z_Nx6, I_N], axis=-1)
        X = self.math.concatenate([top, bottom], axis=-2)

        B_H_I = self.math.homogeneous_inverse(base_transform)
        B_H_I_deriv = adj_derivative(B_H_I, -B_v_C)

        Z_NxN = self.math.factory.zeros(batch_size + (self.NDoF, self.NDoF))
        topd = self.math.concatenate([B_H_I_deriv, Z_6xN], axis=-1)
        bottomd = self.math.concatenate([Z_Nx6, Z_NxN], axis=-1)
        X_dot = self.math.concatenate([topd, bottomd], axis=-2)

        return (I_X_L_dot @ J @ X) + (I_X_L @ J_dot @ X) + (I_X_L @ J @ X_dot)

    def CoM_position(
        self, base_transform: npt.ArrayLike, joint_positions: npt.ArrayLike
    ) -> npt.ArrayLike:
        """Returns the CoM position

        Args:
            base_transform (npt.ArrayLike): The homogenous transform from base to world frame
            joint_positions (npt.ArrayLike): The joints position

        Returns:
            com (npt.ArrayLike): The CoM position
        """
        base_transform, joint_positions = self._convert_to_arraylike(
            base_transform, joint_positions
        )
        batch_size = base_transform.shape[:-2] if len(base_transform.shape) > 2 else ()

        root_to_link = self._compute_tree_homogeneous_traversal(
            base_transform=base_transform,
            joint_positions=joint_positions,
        )
        com_pos = self.math.factory.zeros(batch_size + (3,))
        for idx, node in enumerate(self.model.tree):
            link = node.link
            I_H_l = root_to_link[idx]
            I_H_com = I_H_l @ link.homogeneous()
            link_pos = I_H_com[..., :3, 3:4]
            com_pos += self.math.vxs(link_pos, link.inertial.mass)
        com_pos /= self._convert_to_arraylike(self.get_total_mass())
        return com_pos

    def CoM_jacobian(
        self, base_transform: npt.ArrayLike, joint_positions: npt.ArrayLike
    ) -> npt.ArrayLike:
        """Returns the center of mass (CoM) Jacobian using the centroidal momentum matrix.

        Args:
            base_transform (npt.ArrayLike): The homogenous transform from base to world frame
            joint_positions (npt.ArrayLike): The joints position

        Returns:
            J_com (npt.ArrayLike): The CoM Jacobian
        """
        base_transform, joint_positions = self._convert_to_arraylike(
            base_transform, joint_positions
        )
        batch_size = base_transform.shape[:-2] if len(base_transform.shape) > 2 else ()
        ori_frame_velocity_representation = self.frame_velocity_representation
        self.frame_velocity_representation = Representations.MIXED_REPRESENTATION
        _, Jcm = self.crba(base_transform, joint_positions)
        Xm = self.math.adjoint_mixed(base_transform)
        In = self.math.factory.eye(batch_size + (self.NDoF,))
        Z6n = self.math.factory.zeros(batch_size + (6, self.NDoF))
        Zn6 = self.math.factory.zeros(batch_size + (self.NDoF, 6))
        if (
            ori_frame_velocity_representation
            == Representations.BODY_FIXED_REPRESENTATION
        ):

            top = self.math.concatenate([Xm, Z6n], axis=-1)
            bot = self.math.concatenate([Zn6, In], axis=-1)
            X = self.math.concatenate([top, bot], axis=-2)
            Jcm = Jcm @ X
        elif (
            ori_frame_velocity_representation
            == Representations.INERTIAL_FIXED_REPRESENTATION
        ):
            Xi = self.math.adjoint_inverse(base_transform)
            A = self.math.mtimes(Xm, Xi)
            top = self.math.concatenate([A, Z6n], axis=-1)
            bot = self.math.concatenate([Zn6, In], axis=-1)
            X = self.math.concatenate([top, bot], axis=-2)
            Jcm = Jcm @ X
        self.frame_velocity_representation = ori_frame_velocity_representation
        return Jcm[..., :3, :] / self._convert_to_arraylike(self.get_total_mass())

    def get_total_mass(self):
        """Returns the total mass of the robot

        Returns:
            mass: The total mass
        """
        return self.model.get_total_mass()

    def rnea(
        self,
        base_transform: npt.ArrayLike,
        joint_positions: npt.ArrayLike,
        base_velocity: npt.ArrayLike,
        joint_velocities: npt.ArrayLike,
        g: npt.ArrayLike,
    ) -> npt.ArrayLike:
        """
        Batched Recursive Newton-Euler (reduced: no joint/base accelerations, no external forces).

        Args:
            base_transform (npt.ArrayLike): The homogenous transform from base to world frame
            joint_positions (npt.ArrayLike): The joints position
            base_velocity (npt.ArrayLike): The base spatial velocity
            joint_velocities (npt.ArrayLike): The joints velocities
            g (npt.ArrayLike): The gravity vector

        Returns:
            tau (npt.ArrayLike): The vector of generalized forces
        """
        base_transform, joint_positions, base_velocity, joint_velocities, g = (
            self._convert_to_arraylike(
                base_transform, joint_positions, base_velocity, joint_velocities, g
            )
        )

        math = self.math
        n = self.NDoF
        node_count = self._node_count
        parent_indices = self._parent_indices
        joint_indices = self._joint_indices_per_node
        motion_subspaces = self._motion_subspaces
        inertias = self._spatial_inertias
        joints = self._joints_per_node
        root_idx = self._root_index
        node_indices = self._node_indices
        rev_indices = self._rev_node_indices

        batch_shape = (
            tuple(base_transform.shape[:-2]) if base_transform.ndim > 2 else ()
        )

        if (
            self.frame_velocity_representation
            == Representations.INERTIAL_FIXED_REPRESENTATION
        ):
            gravity_X = math.adjoint_inverse(base_transform)
        else:
            gravity_X = math.adjoint_mixed_inverse(base_transform)  # (...,6,6)

        if (
            self.frame_velocity_representation
            == Representations.BODY_FIXED_REPRESENTATION
        ):
            # frame C = B for body-fixed representation
            B_X_C = math.factory.eye(batch_shape + (6,))
            transformed_acc = math.factory.zeros(batch_shape + (6,))
        elif self.frame_velocity_representation == Representations.MIXED_REPRESENTATION:
            # frame C = B[I] for mixed representation
            B_X_C = math.adjoint_mixed_inverse(base_transform)
            omega = base_velocity[..., 3:]
            vlin = base_velocity[..., :3]
            skew_omega_times_vlin = math.mxv(math.skew(omega), vlin)
            top3 = -math.mxv(B_X_C[..., :3, :3], skew_omega_times_vlin)
            bot3 = math.factory.zeros(batch_shape + (3,))
            transformed_acc = math.concatenate([top3, bot3], axis=-1)
        elif (
            self.frame_velocity_representation
            == Representations.INERTIAL_FIXED_REPRESENTATION
        ):
            # frame C = I for inertial-fixed representation
            B_X_C = math.adjoint_inverse(base_transform)
            transformed_acc = math.factory.zeros(batch_shape + (6,))
        else:
            raise NotImplementedError(
                "Only BODY_FIXED_REPRESENTATION, MIXED_REPRESENTATION and INERTIAL_FIXED_REPRESENTATION are implemented"
            )

        a0 = -(math.mxv(gravity_X, g)) + transformed_acc

        if n > 0:
            zero_q = math.zeros_like(joint_positions[..., 0])
            zero_qd = math.zeros_like(joint_velocities[..., 0])
        else:
            zero_q = math.zeros_like(base_velocity[..., 0])
            zero_qd = zero_q

        Ic = [None] * node_count
        Xup = [None] * node_count
        v = [None] * node_count
        a = [None] * node_count
        f = [None] * node_count

        for idx in node_indices:
            Ic[idx] = inertias[idx]

            if idx == root_idx:
                Xup[idx] = self._root_spatial_transform
                v[idx] = math.mxv(B_X_C, base_velocity)
                a[idx] = math.mxv(Xup[idx], a0)
                continue

            joint = joints[idx]
            parent = parent_indices[idx]
            joint_idx = joint_indices[idx]

            q = joint_positions[..., joint_idx] if joint_idx is not None else zero_q
            qd = joint_velocities[..., joint_idx] if joint_idx is not None else zero_qd

            X = joint.spatial_transform(q=q)
            Xup[idx] = X

            Phi_i = motion_subspaces[idx]
            phi_qd = math.vxs(Phi_i, qd)
            v[idx] = math.mxv(X, v[parent]) + phi_qd
            a[idx] = math.mxv(X, a[parent]) + math.mxv(
                math.spatial_skew(v[idx]), phi_qd
            )

            f[idx] = math.mxv(Ic[idx], a[idx]) + math.mxv(
                math.spatial_skew_star(v[idx]), math.mxv(Ic[idx], v[idx])
            )

        # Root wrench contribution (skipped in loop above)
        f[root_idx] = math.mxv(Ic[root_idx], a[root_idx]) + math.mxv(
            math.spatial_skew_star(v[root_idx]), math.mxv(Ic[root_idx], v[root_idx])
        )

        tau_base = None
        tau_joint_cols = [None] * n if n > 0 else []

        for idx in rev_indices:
            Phi_i = motion_subspaces[idx]
            Fi = f[idx]
            Phi_T = math.swapaxes(Phi_i, -2, -1)

            if idx == root_idx:
                tau_base = math.mxv(Phi_T, Fi)
            else:
                joint_idx = joint_indices[idx]
                if joint_idx is not None:
                    tau_val = math.mxv(Phi_T, Fi)
                    self._scatter_entries(tau_joint_cols, joint_idx, tau_val)
                parent = parent_indices[idx]
                if parent >= 0:
                    f[parent] = f[parent] + math.mxv(
                        math.swapaxes(Xup[idx], -2, -1), Fi
                    )

        tau_base = math.mxv(math.swapaxes(B_X_C, -2, -1), tau_base)

        if n > 0:
            zero_tau = math.factory.zeros(batch_shape + (1,))
            tau_joints = [
                col if col is not None else zero_tau for col in tau_joint_cols
            ]
            tau_joints_vec = math.concatenate(tau_joints, axis=-1)
        else:
            tau_joints_vec = math.factory.zeros(batch_shape + (0,))

        return math.concatenate([tau_base, tau_joints_vec], axis=-1)

    def aba(
        self,
        base_transform: npt.ArrayLike,
        joint_positions: npt.ArrayLike,
        base_velocity: npt.ArrayLike,
        joint_velocities: npt.ArrayLike,
        joint_torques: npt.ArrayLike,
        g: npt.ArrayLike,
        external_wrenches: dict[str, npt.ArrayLike] | None = None,
    ) -> npt.ArrayLike:
        """Featherstone Articulated Body Algorithm for floating-base forward dynamics.

        Args:
            base_transform (npt.ArrayLike): The homogenous transform from base to world frame
            joint_positions (npt.ArrayLike): The joints position
            base_velocity (npt.ArrayLike): The spatial velocity of the base
            joint_velocities (npt.ArrayLike): The joints velocities
            joint_torques (npt.ArrayLike): The joints torques/forces
            g (npt.ArrayLike | None, optional): The gravity vector
            external_wrenches (dict[str, npt.ArrayLike] | None, optional): A dictionary of external wrenches applied to specific links. Keys are link names, and values are 6D wrench vectors. Defaults to None.

        Returns:
            accelerations (npt.ArrayLike): The spatial acceleration of the base and joints accelerations
        """
        math = self.math
        n = self.NDoF
        node_count = self._node_count
        parent_indices = self._parent_indices
        joint_indices = self._joint_indices_per_node
        motion_subspaces = self._motion_subspaces
        inertias = self._spatial_inertias
        joints = self._joints_per_node
        root_idx = self._root_index
        node_indices = self._node_indices
        rev_indices = self._rev_node_indices

        (
            base_transform,
            joint_positions,
            base_velocity,
            joint_velocities,
            joint_torques,
            g,
        ) = self._convert_to_arraylike(
            base_transform,
            joint_positions,
            base_velocity,
            joint_velocities,
            joint_torques,
            g,
        )

        batch_shape = (
            tuple(base_transform.shape[:-2]) if base_transform.ndim > 2 else ()
        )

        if external_wrenches is not None:
            generalized_ext = math.factory.zeros(batch_shape + (6 + n,))
            for frame, wrench in external_wrenches.items():
                wrench_arr = self._convert_to_arraylike(wrench)
                J = self.jacobian(frame, base_transform, joint_positions)
                generalized_ext = generalized_ext + math.mxv(
                    math.swapaxes(J, -2, -1), wrench_arr
                )

            base_ext = generalized_ext[..., :6]
            joint_ext = (
                generalized_ext[..., 6:] if n > 0 else math.zeros_like(joint_torques)
            )
        else:
            base_ext = math.factory.zeros(batch_shape + (6,))
            joint_ext = math.zeros_like(joint_torques)

        joint_torques_eff = joint_torques + joint_ext

        if self.frame_velocity_representation == Representations.MIXED_REPRESENTATION:
            B_X_I = math.adjoint_mixed_inverse(base_transform)
            I_X_B = math.adjoint_mixed(base_transform)
        elif (
            self.frame_velocity_representation
            == Representations.INERTIAL_FIXED_REPRESENTATION
        ):
            B_X_I = math.adjoint_inverse(base_transform)
            I_X_B = math.adjoint(base_transform)
        elif (
            self.frame_velocity_representation
            == Representations.BODY_FIXED_REPRESENTATION
        ):
            B_X_I = I_X_B = math.factory.eye(batch_shape + (6,))
        else:
            raise NotImplementedError(
                "Only BODY_FIXED_REPRESENTATION, MIXED_REPRESENTATION and INERTIAL_FIXED_REPRESENTATION are implemented"
            )

        base_velocity_body = math.mxv(B_X_I, base_velocity)

        B_star_BI = math.swapaxes(I_X_B, -2, -1)
        base_ext_body = math.mxv(B_star_BI, base_ext)

        a0_input = math.mxv(math.adjoint_mixed_inverse(base_transform), g)

        def zeros6():
            return math.factory.zeros(batch_shape + (6,))

        def eye6():
            return math.factory.eye(batch_shape + (6,))

        def expand_to_match(vec, reference):
            expanded = math.expand_dims(vec, axis=-1)
            expanded_ndim = expanded.ndim
            reference_ndim = reference.ndim
            if expanded_ndim != reference_ndim:
                expanded = math.expand_dims(expanded, axis=-1)
            return expanded

        if n > 0:
            zero_q = math.zeros_like(joint_positions[..., 0])
        else:
            zero_q = math.zeros_like(base_velocity[..., 0])

        def tile_batch(arr):
            if not batch_shape:
                return arr
            reps = batch_shape + (1,) * len(arr.shape)
            return math.tile(arr, reps)

        Xup = [None] * node_count
        Scols: list[ArrayLike | None] = [None] * node_count
        v = [None] * node_count
        c = [None] * node_count
        IA = [None] * node_count
        pA = [None] * node_count
        g_acc = [None] * node_count

        for idx in node_indices:
            IA[idx] = tile_batch(inertias[idx])

            if idx == root_idx:
                Xup[idx] = eye6()
                v[idx] = base_velocity_body
                c[idx] = zeros6()
                g_acc[idx] = a0_input
            else:
                parent = parent_indices[idx]
                joint = joints[idx]
                joint_idx = joint_indices[idx]
                q_i = (
                    joint_positions[..., joint_idx] if joint_idx is not None else zero_q
                )
                X_current = joint.spatial_transform(q=q_i)
                Xup[idx] = X_current

                g_acc[idx] = math.mxv(X_current, g_acc[parent])

                if joint_idx is not None:
                    Si = tile_batch(motion_subspaces[idx])
                    Scols[idx] = Si
                    qd_i = joint_velocities[..., joint_idx]
                    vJ = math.vxs(Si, qd_i)
                else:
                    Scols[idx] = None
                    vJ = zeros6()

                v[idx] = math.mxv(X_current, v[parent]) + vJ
                c[idx] = math.mxv(math.spatial_skew(v[idx]), vJ)

            pA[idx] = math.mxv(
                math.spatial_skew_star(v[idx]), math.mxv(IA[idx], v[idx])
            )

        d_list: list[ArrayLike | None] = [None] * node_count
        inv_d_list: list[ArrayLike | None] = [None] * node_count
        u_list: list[ArrayLike | None] = [None] * node_count
        U_list: list[ArrayLike | None] = [None] * node_count

        for idx in rev_indices:
            if idx == root_idx:
                continue

            parent = parent_indices[idx]
            Xpt = math.swapaxes(Xup[idx], -2, -1)
            Si = Scols[idx]
            if Scols[idx] is not None:
                U_i = math.mtimes(IA[idx], Si)
                d_i = math.mtimes(math.swapaxes(Si, -2, -1), U_i)
                joint_idx = joint_indices[idx]
                if isinstance(joint_idx, tuple):
                    tau_slice = slice(joint_idx[0], joint_idx[-1] + 1)
                else:
                    tau_slice = slice(joint_idx, joint_idx + 1)
                tau_vec = joint_torques_eff[..., tau_slice]
                Si_T_pA = math.mxv(math.swapaxes(Si, -2, -1), pA[idx])
                u_i = tau_vec - Si_T_pA

                d_list[idx] = d_i
                u_list[idx] = u_i
                U_list[idx] = U_i

                inv_d = math.inv(d_i)
                inv_d_list[idx] = inv_d
                Ia = IA[idx] - math.mtimes(
                    U_i, math.mtimes(inv_d, math.swapaxes(U_i, -2, -1))
                )
                gain = math.mtimes(inv_d, expand_to_match(u_i, inv_d))
                gain_vec = gain[..., 0]
                pa = pA[idx] + math.mxv(Ia, c[idx]) + math.mxv(U_i, gain_vec)
            else:
                Ia = IA[idx]
                pa = pA[idx] + math.mxv(Ia, c[idx])

            IA[parent] = IA[parent] + math.mtimes(math.mtimes(Xpt, Ia), Xup[idx])
            pA[parent] = pA[parent] + math.mxv(Xpt, pa)

        rhs_root = base_ext_body - pA[root_idx] + math.mxv(IA[root_idx], a0_input)
        a_base = math.solve(IA[root_idx], rhs_root)

        a = [None] * node_count
        a[root_idx] = a_base

        qdd_entries: list[ArrayLike | None] = [None] * n if n > 0 else []

        for idx in node_indices:
            if idx == root_idx:
                continue

            parent = parent_indices[idx]
            a_pre = math.mxv(Xup[idx], a[parent]) + c[idx]
            free_acc = g_acc[idx]
            rel_acc = a_pre - free_acc if free_acc is not None else a_pre

            Si = Scols[idx]
            joint_idx = joint_indices[idx]

            if Si is not None and joint_idx is not None:
                U_i = U_list[idx]
                U_T_rel_acc = math.mxv(math.swapaxes(U_i, -2, -1), rel_acc)
                num = u_list[idx] - U_T_rel_acc
                inv_d = inv_d_list[idx]
                num_expanded = expand_to_match(num, inv_d)
                gain_qdd = math.mtimes(inv_d, num_expanded)
                qdd_col = gain_qdd[..., 0]
                max_idx = max(joint_idx) if isinstance(joint_idx, tuple) else joint_idx
                if max_idx < n:
                    self._scatter_entries(qdd_entries, joint_idx, qdd_col)
                a_correction_vec = math.mxv(Si, qdd_col)
                a[idx] = a_pre + a_correction_vec
            else:
                a[idx] = a_pre

        if n > 0:
            zero_col = math.factory.zeros(batch_shape + (1,))
            qdd_cols = [
                entry if entry is not None else zero_col for entry in qdd_entries
            ]
            joint_qdd = math.concatenate(qdd_cols, axis=-1)
        else:
            joint_qdd = math.factory.zeros(batch_shape + (0,))

        if self.frame_velocity_representation == Representations.MIXED_REPRESENTATION:
            Xm = math.adjoint_mixed(base_transform)
            base_vel_mixed = math.mxv(Xm, base_velocity_body)
            Xm_dot = math.adjoint_mixed_derivative(base_transform, base_vel_mixed)
            base_acc = math.mxv(Xm, a_base) + math.mxv(Xm_dot, base_velocity_body)
        elif (
            self.frame_velocity_representation
            == Representations.INERTIAL_FIXED_REPRESENTATION
        ):
            X = math.adjoint(base_transform)
            base_vel_inertial = math.mxv(X, base_velocity_body)
            X_dot = math.adjoint_derivative(base_transform, base_vel_inertial)
            base_acc = math.mxv(X, a_base) + math.mxv(X_dot, base_velocity_body)
        else:
            base_acc = a_base

        return math.concatenate([base_acc, joint_qdd], axis=-1)

    def _get_joints_chain_cached(self, frame: str) -> tuple[Joint, ...]:
        """Return the root->frame chain and cache its static joint sequence."""
        cached_chain = self._joint_chain_cache.get(frame)
        if cached_chain is not None:
            return cached_chain

        if frame not in self.model.links and frame not in self.model.frames:
            raise ValueError(f"{frame} is not in the robot model.")

        chain_reversed: list[Joint] = []
        current = frame
        while current != self.root_link:
            joint = self._joint_by_child.get(current)
            if joint is None:
                raise ValueError(
                    f"Unable to find a chain from {self.root_link} to {frame}."
                )
            chain_reversed.append(joint)
            current = joint.parent

        chain = tuple(reversed(chain_reversed))
        self._joint_chain_cache[frame] = chain
        return chain

    def _compute_chain_homogeneous_traversal(
        self,
        frame: str,
        base_transform: npt.ArrayLike,
        joint_positions: npt.ArrayLike,
    ) -> ChainTraversal:
        """Traverse root->frame once and cache every intermediate transform."""
        chain = self._get_joints_chain_cached(frame)
        if not chain:
            return ChainTraversal(
                joints=chain,
                root_to_joint=(),
                root_to_target=base_transform,
            )

        zero_q = self._default_joint_value(
            joint_positions=joint_positions,
            fallback=base_transform[..., 0, 0],
        )
        current = base_transform
        root_to_joint: list[ArrayLike] = [None] * len(chain)
        for idx, joint in enumerate(chain):
            q = joint_positions[..., joint.idx] if joint.idx is not None else zero_q
            current = current @ joint.homogeneous(q=q)
            root_to_joint[idx] = current

        return ChainTraversal(
            joints=chain,
            root_to_joint=tuple(root_to_joint),
            root_to_target=current,
        )

    def _compute_tree_homogeneous_traversal(
        self,
        base_transform: npt.ArrayLike,
        joint_positions: npt.ArrayLike,
    ) -> tuple[ArrayLike, ...]:
        """Traverse the full tree once and cache root->link transforms."""
        root_to_link: list[ArrayLike] = [None] * self._node_count
        root_to_link[self._root_index] = base_transform

        zero_q = self._default_joint_value(
            joint_positions=joint_positions,
            fallback=base_transform[..., 0, 0],
        )
        for idx in self._node_indices:
            if idx == self._root_index:
                continue

            parent_idx = self._parent_indices[idx]
            joint = self._joints_per_node[idx]
            if joint is None:
                root_to_link[idx] = root_to_link[parent_idx]
                continue

            q = joint_positions[..., joint.idx] if joint.idx is not None else zero_q
            root_to_link[idx] = root_to_link[parent_idx] @ joint.homogeneous(q=q)

        return tuple(root_to_link)

    def _joints_jacobian_from_traversal(
        self,
        traversal: ChainTraversal,
        joint_positions: npt.ArrayLike,
    ) -> npt.ArrayLike:
        """Build a joint Jacobian from cached chain transforms."""
        batch_size = (
            joint_positions.shape[:-1] if len(joint_positions.shape) >= 2 else ()
        )
        if self.NDoF == 0:
            return self.math.factory.zeros(batch_size + (6, 0))

        L_H_B = self.math.homogeneous_inverse(traversal.root_to_target)
        cols: list = [None] * self.NDoF
        for joint, B_H_j in zip(traversal.joints, traversal.root_to_joint):
            if joint.idx is None:
                continue
            L_H_j = L_H_B @ B_H_j
            S = joint.motion_subspace()
            J_block = self.math.adjoint(L_H_j) @ S
            self._scatter_cols(cols, joint.idx, J_block)

        zero_col = self.math.factory.zeros(batch_size + (6, 1))
        cols = [zero_col if col is None else col for col in cols]
        return self.math.concatenate(cols, axis=-1)

    @staticmethod
    def _scatter_cols(
        cols: list, idx: int | tuple[int, ...], block: npt.ArrayLike
    ) -> None:
        """Assign columns of *block* to *cols* at the DOF positions given by *idx*."""
        if isinstance(idx, tuple):
            for i, j in enumerate(idx):
                cols[j] = block[..., :, i : i + 1]
        else:
            cols[idx] = block

    @staticmethod
    def _scatter_entries(
        entries: list, idx: int | tuple[int, ...], vec: npt.ArrayLike
    ) -> None:
        """Assign elements of *vec* to *entries* at the DOF positions given by *idx*."""
        if isinstance(idx, tuple):
            for i, j in enumerate(idx):
                entries[j] = vec[..., i : i + 1]
        else:
            entries[idx] = vec

    def _default_joint_value(
        self, joint_positions: npt.ArrayLike, fallback: npt.ArrayLike | None = None
    ) -> npt.ArrayLike:
        """Return a zero scalar/vector matching the joint position batch shape."""
        if joint_positions.shape[-1] > 0:
            return self.math.zeros_like(joint_positions[..., 0])
        if fallback is not None:
            return self.math.zeros_like(fallback)
        if len(joint_positions.shape) >= 2:
            return self.math.factory.zeros(joint_positions.shape[:-1])
        return self.math.factory.zeros(())

    def _convert_to_arraylike(self, *args):
        """Convert inputs to ArrayLike if they are not already.
        Args:
            *args: Input arguments to be converted.
        Returns:
            Converted arguments as ArrayLike.
        """
        if not args:
            raise ValueError("At least one argument is required")

        converted = []
        for arg in args:
            if isinstance(arg, ArrayLike):
                converted.append(arg)
            else:
                converted.append(self.math.asarray(arg))
        return converted[0] if len(converted) == 1 else converted

    def _prepare_tree_cache(self) -> None:
        """Pre-compute static tree data so the dynamic algorithms avoid repeated Python work."""
        nodes = list(self.model.tree)
        node_count = len(nodes)
        self._node_indices = tuple(range(node_count))
        self._rev_node_indices = tuple(reversed(self._node_indices))
        self._parent_indices = [-1] * node_count
        self._joint_indices_per_node: list[int | tuple[int, ...] | None] = [
            None
        ] * node_count
        self._motion_subspaces: list[npt.ArrayLike] = [None] * node_count
        self._spatial_inertias: list[npt.ArrayLike] = [None] * node_count
        self._joints_per_node: list[Joint | None] = [None] * node_count
        self._joint_index_to_node: dict[int, int] = {}
        self._joint_index_to_local_dof: dict[int, int] = {}
        self._joint_by_child: dict[str, Joint] = {}
        self._joint_chain_cache: dict[str, tuple[Joint, ...]] = {
            self.root_link: tuple()
        }

        for idx, node in enumerate(nodes):
            link, joint, parent_link = node.get_elements()
            self._joints_per_node[idx] = joint
            self._spatial_inertias[idx] = link.spatial_inertia()
            parent_idx = (
                self.model.tree.get_idx_from_name(parent_link.name)
                if parent_link is not None
                else -1
            )
            self._parent_indices[idx] = parent_idx
            if joint is None:
                self._motion_subspaces[idx] = self._root_motion_subspace
                self._joint_indices_per_node[idx] = None
            else:
                self._motion_subspaces[idx] = joint.motion_subspace()
                self._joint_indices_per_node[idx] = joint.idx
                if joint.idx is not None:
                    if isinstance(joint.idx, tuple):
                        for local, global_idx in enumerate(joint.idx):
                            self._joint_index_to_node[global_idx] = idx
                            self._joint_index_to_local_dof[global_idx] = local
                    else:
                        self._joint_index_to_node[int(joint.idx)] = idx
                        self._joint_index_to_local_dof[int(joint.idx)] = 0

        for joint in self.model.joints.values():
            self._joint_by_child[joint.child] = joint

        self._root_index = self.model.tree.get_idx_from_name(self.root_link)
        self._node_count = node_count
