import abc
import dataclasses

import numpy.typing as npt

from adam.core.spatial_math import SpatialMath


@dataclasses.dataclass(frozen=True, slots=True)
class Pose:
    """Pose class"""

    xyz: npt.ArrayLike
    rpy: npt.ArrayLike

    @staticmethod
    def build(xyz: npt.ArrayLike, rpy: npt.ArrayLike, math: SpatialMath) -> "Pose":
        xyz = math.asarray(xyz)
        rpy = math.asarray(rpy)
        return Pose(xyz, rpy)

    @staticmethod
    def zero(math: SpatialMath) -> "Pose":
        return Pose.build([0, 0, 0], [0, 0, 0], math)

    def get_xyz(self) -> npt.ArrayLike:
        return self.xyz

    def get_rpy(self) -> npt.ArrayLike:
        return self.rpy


@dataclasses.dataclass(frozen=True, slots=True)
class Inertia:
    matrix: npt.ArrayLike
    ixx: npt.DTypeLike
    ixy: npt.DTypeLike
    ixz: npt.DTypeLike
    iyy: npt.DTypeLike
    iyz: npt.DTypeLike
    izz: npt.DTypeLike

    @staticmethod
    def build(
        ixx: npt.DTypeLike,
        ixy: npt.DTypeLike,
        ixz: npt.DTypeLike,
        iyy: npt.DTypeLike,
        iyz: npt.DTypeLike,
        izz: npt.DTypeLike,
        math: SpatialMath,
    ) -> "Inertia":
        matrix = math.asarray(
            [
                [ixx, ixy, ixz],
                [ixy, iyy, iyz],
                [ixz, iyz, izz],
            ]
        )
        return Inertia(matrix, ixx, ixy, ixz, iyy, iyz, izz)

    @staticmethod
    def zero(math: SpatialMath) -> "Inertia":
        return Inertia.build(
            ixx=0.0, ixy=0.0, ixz=0.0, iyy=0.0, iyz=0.0, izz=0.0, math=math
        )

    def get_matrix(self) -> npt.ArrayLike:
        return self.matrix


@dataclasses.dataclass(frozen=True, slots=True)
class Limits:
    """Limits class"""

    lower: npt.ArrayLike
    upper: npt.ArrayLike
    effort: npt.ArrayLike
    velocity: npt.ArrayLike


@dataclasses.dataclass
class Joint(abc.ABC):
    """Base Joint class. You need to fill at least these fields"""

    math: SpatialMath
    name: str
    parent: str
    child: str
    type: str
    axis: npt.ArrayLike
    origin: Pose
    limit: Limits
    idx: int | tuple[int, ...]
    dofs: int
    """
    Abstract base class for all joints.
    """

    @abc.abstractmethod
    def spatial_transform(self, q: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            q (npt.ArrayLike): joint motion

        Returns:
            npt.ArrayLike: spatial transform of the joint given q
        """
        pass

    @abc.abstractmethod
    def motion_subspace(self) -> npt.ArrayLike:
        """
        Returns:
            npt.ArrayLike: motion subspace of the joint
        """

    @abc.abstractmethod
    def homogeneous(self, q: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            q (npt.ArrayLike): joint value
        Returns:
            npt.ArrayLike: homogeneous transform given the joint value
        """
        pass


@dataclasses.dataclass(frozen=True, slots=True)
class Inertial:
    """Inertial description"""

    mass: npt.DTypeLike
    inertia: Inertia
    origin: Pose

    @staticmethod
    def zero(math: SpatialMath) -> "Inertial":
        """Returns an Inertial object with zero mass and inertia"""
        xyz = math.factory.zeros(3)
        rpy = math.factory.zeros(3)
        zero_element = math.factory.zeros(1)
        return Inertial(
            mass=zero_element,
            inertia=Inertia.zero(math),
            origin=Pose(xyz=xyz, rpy=rpy),
        )

    def set_mass(self, mass: npt.ArrayLike) -> "Inertial":
        """Set the mass of the inertial object"""
        self.mass = mass

    def set_inertia(self, inertia: Inertia) -> "Inertial":
        """Set the inertia of the inertial object"""
        self.inertia = inertia

    def set_origin(self, origin: Pose) -> "Inertial":
        """Set the origin of the inertial object"""
        self.origin = origin


@dataclasses.dataclass
class Link(abc.ABC):
    """Base Link class. You need to fill at least these fields"""

    math: SpatialMath
    name: str
    visuals: list
    inertial: Inertial
    collisions: list

    @abc.abstractmethod
    def spatial_inertia(self) -> npt.ArrayLike:
        """
        Returns:
            npt.ArrayLike: the 6x6 inertia matrix expressed at
                           the origin of the link (with rotation)
        """
        pass

    @abc.abstractmethod
    def homogeneous(self) -> npt.ArrayLike:
        """
        Returns:
            npt.ArrayLike: the homogeneous transform of the link
        """
        pass


@dataclasses.dataclass
class ModelFactory(abc.ABC):
    """The abstract class of the model factory.

    The model factory is responsible for creating the model.

    You need to implement all the methods in your concrete implementation
    """

    math: SpatialMath
    name: str

    @abc.abstractmethod
    def build_link(self) -> Link:
        """build the single link
        Returns:
            Link
        """
        pass

    @abc.abstractmethod
    def build_joint(self) -> Joint:
        """build the single joint

        Returns:
            Joint
        """
        pass

    @abc.abstractmethod
    def get_links(self) -> list[Link]:
        """
        Returns:
            list[Link]: the list of the link
        """
        pass

    @abc.abstractmethod
    def get_frames(self) -> list[Link]:
        """
        Returns:
            list[Link]: the list of the frames
        """
        pass

    @abc.abstractmethod
    def get_joints(self) -> list[Joint]:
        """
        Returns:
            list[Joint]: the list of the joints
        """
        pass
