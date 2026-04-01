import idyntree.bindings
import numpy as np
import urdf_parser_py.urdf
import casadi as cs

from adam.core.spatial_math import ArrayLike, SpatialMath
from adam.model.abc_factories import Joint, Link
from adam.model.model import Model
from adam.model.std_factories.std_joint import StdJoint
from adam.model.std_factories.std_link import StdLink


def _to_sequence(x) -> list[float]:
    """Coerce array-like objects to a plain Python list of floats.

    Supports:
    - CasADi DM, SX, MX (uses .full() when available)
    - Objects exposing an `array` attribute (e.g., CasadiLike wrapper)
    - numpy arrays, lists, tuples and other iterables
    - scalars
    """
    # Unwrap wrapper that stores the underlying array in `.array`
    val = x.array if isinstance(x, ArrayLike) else x
    if isinstance(val, (cs.DM, cs.SX, cs.MX)):
        # Convert to full() representation and flatten to 1D list
        dm_full = cs.DM(val).full()
        val = [float(v) for row in dm_full for v in row]
        return val

    for i, v in enumerate(val):
        if isinstance(v, ArrayLike):
            val[i] = (
                cs.DM(v).full() if isinstance(v, (cs.DM, cs.SX, cs.MX)) else v.array
            )
    # Handle CasADi types if available. It should be already a casadi type, but let's be safe
    val = cs.DM(val).full() if isinstance(val, (cs.DM)) else val
    return [float(v) for v in val]


def _to_scalar(x) -> float:
    """Coerce a scalar-like object to float (supports CasADi and wrappers)."""
    # Unwrap wrapper that stores the underlying array in `.array`
    val = x.array if isinstance(x, ArrayLike) else x
    # Handle CasADi types if available. It should be already a casadi type, but let's be safe
    if isinstance(val, (cs.DM, cs.SX, cs.MX)):
        dm_full = cs.DM(val).full()
        # `full()` returns a NumPy array; flatten and extract the single scalar
        val = dm_full.flat[0]
    return float(val)


def to_idyntree_solid_shape(
    visual: urdf_parser_py.urdf.Visual,
) -> idyntree.bindings.SolidShape:
    """
    Args:
        visual (urdf_parser_py.urdf.Visual): the visual to convert

    Returns:
        iDynTree.SolidShape: the iDynTree solid shape
    """
    visual_position = idyntree.bindings.Position.FromPython(
        _to_sequence(visual.origin.xyz)
    )
    visual_rotation = idyntree.bindings.Rotation.RPY(*_to_sequence(visual.origin.rpy))
    visual_transform = idyntree.bindings.Transform()
    visual_transform.setRotation(visual_rotation)
    visual_transform.setPosition(visual_position)
    material = idyntree.bindings.Material(visual.material.name)
    if visual.material.color is not None:
        color = idyntree.bindings.Vector4()
        color[0] = visual.material.color.rgba[0]
        color[1] = visual.material.color.rgba[1]
        color[2] = visual.material.color.rgba[2]
        color[3] = visual.material.color.rgba[3]
        material.setColor(color)
    if isinstance(visual.geometry, urdf_parser_py.urdf.Box):
        output = idyntree.bindings.Box()
        output.setX(visual.geometry.size[0])
        output.setY(visual.geometry.size[1])
        output.setZ(visual.geometry.size[2])
        output.setLink_H_geometry(visual_transform)
        return output
    if isinstance(visual.geometry, urdf_parser_py.urdf.Cylinder):
        output = idyntree.bindings.Cylinder()
        output.setRadius(visual.geometry.radius)
        output.setLength(visual.geometry.length)
        output.setLink_H_geometry(visual_transform)
        return output
    if isinstance(visual.geometry, urdf_parser_py.urdf.Sphere):
        output = idyntree.bindings.Sphere()
        output.setRadius(visual.geometry.radius)
        output.setLink_H_geometry(visual_transform)
        return output
    if isinstance(visual.geometry, urdf_parser_py.urdf.Mesh):
        output = idyntree.bindings.ExternalMesh()
        output.setFilename(visual.geometry.filename)
        output.setScale(visual.geometry.scale)
        output.setLink_H_geometry(visual_transform)
        return output

    raise NotImplementedError(
        f"The visual type {visual.geometry.__class__} is not supported"
    )


def to_idyntree_link(
    link: Link,
) -> tuple[idyntree.bindings.Link, list[idyntree.bindings.SolidShape]]:
    """
    Args:
        link (Link): the link to convert

    Returns:
        A tuple containing the iDynTree link and the iDynTree solid shapes
    """
    output = idyntree.bindings.Link()
    input_inertia = link.inertial.inertia
    inertia_matrix = np.array(
        [
            [input_inertia.ixx, input_inertia.ixy, input_inertia.ixz],
            [input_inertia.ixy, input_inertia.iyy, input_inertia.iyz],
            [input_inertia.ixz, input_inertia.iyz, input_inertia.izz],
        ]
    )
    inertia_rotation = idyntree.bindings.Rotation.RPY(
        *_to_sequence(link.inertial.origin.rpy)
    )
    idyn_spatial_rotational_inertia = idyntree.bindings.RotationalInertia()
    for i in range(3):
        for j in range(3):
            idyn_spatial_rotational_inertia.setVal(i, j, inertia_matrix[i, j])
    rotated_inertia = inertia_rotation * idyn_spatial_rotational_inertia
    idyn_spatial_inertia = idyntree.bindings.SpatialInertia()
    com_position = idyntree.bindings.Position.FromPython(
        _to_sequence(link.inertial.origin.xyz)
    )
    idyn_spatial_inertia.fromRotationalInertiaWrtCenterOfMass(
        _to_scalar(link.inertial.mass),
        com_position,
        rotated_inertia,
    )
    output.setInertia(idyn_spatial_inertia)

    return output, [to_idyntree_solid_shape(v) for v in link.visuals]


def to_idyntree_joint(
    joint: Joint, parent_index: int, child_index: int
) -> idyntree.bindings.IJoint:
    """
    Args:
        joint (Joint): the joint to convert
        parent_index (int): the parent link index
        child_index (int): the child link index
    Returns:
        iDynTree.bindings.IJoint: the iDynTree joint
    """

    rest_position = idyntree.bindings.Position.FromPython(
        _to_sequence(joint.origin.xyz)
    )
    rest_rotation = idyntree.bindings.Rotation.RPY(*_to_sequence(joint.origin.rpy))
    rest_transform = idyntree.bindings.Transform()
    rest_transform.setRotation(rest_rotation)
    rest_transform.setPosition(rest_position)

    if joint.type == "fixed":
        return idyntree.bindings.FixedJoint(parent_index, child_index, rest_transform)

    direction = idyntree.bindings.Direction(*_to_sequence(joint.axis))
    origin = idyntree.bindings.Position.Zero()
    axis = idyntree.bindings.Axis()
    axis.setDirection(direction)
    axis.setOrigin(origin)

    if joint.type in ["revolute", "continuous"]:
        output = idyntree.bindings.RevoluteJoint()
        output.setAttachedLinks(parent_index, child_index)
        output.setRestTransform(rest_transform)
        output.setAxis(axis, child_index, parent_index)
        if joint.limit is not None and joint.type == "revolute":
            output.setPosLimits(
                0, _to_scalar(joint.limit.lower), _to_scalar(joint.limit.upper)
            )
        return output
    if joint.type in ["prismatic"]:
        output = idyntree.bindings.PrismaticJoint()
        output.setAttachedLinks(parent_index, child_index)
        output.setRestTransform(rest_transform)
        output.setAxis(axis, child_index, parent_index)
        if joint.limit is not None:
            output.setPosLimits(
                0, _to_scalar(joint.limit.lower), _to_scalar(joint.limit.upper)
            )
        return output

    NotImplementedError(f"The joint type {joint.type} is not supported")


def to_idyntree_model(model: Model) -> idyntree.bindings.Model:
    """
    Args:
        model (Model): the model to convert

    Returns:
        iDynTree.Model: the iDynTree model
    """

    output = idyntree.bindings.Model()
    output_visuals = []
    links_map = {}

    for node in model.tree:
        link, visuals = to_idyntree_link(node.link)
        link_index = output.addLink(node.name, link)
        assert output.isValidLinkIndex(link_index)
        assert link_index == len(output_visuals)
        output_visuals.append(visuals)
        links_map[node.name] = link_index

    for i, visuals in enumerate(output_visuals):
        output.visualSolidShapes().clearSingleLinkSolidShapes(i)
        for visual in visuals:
            output.visualSolidShapes().addSingleLinkSolidShape(i, visual)

    for node in model.tree:
        for j in node.arcs:
            assert j.name not in model.frames
            if j.type == "spherical":
                parent_idx = links_map[j.parent]
                child_idx = links_map[j.child]

                # Create rest transform from joint origin
                rest_position = idyntree.bindings.Position.FromPython(
                    _to_sequence(j.origin.xyz)
                )
                rest_rotation = idyntree.bindings.Rotation.RPY(
                    *_to_sequence(j.origin.rpy)
                )
                rest_transform = idyntree.bindings.Transform()
                rest_transform.setRotation(rest_rotation)
                rest_transform.setPosition(rest_position)

                # Create SphericalJoint with rest transform
                spherical_joint = idyntree.bindings.SphericalJoint(rest_transform)
                spherical_joint.setAttachedLinks(parent_idx, child_idx)

                # Set the joint center at the joint origin (relative to parent link)
                spherical_joint.setJointCenter(parent_idx, rest_position)

                joint_index = output.addJoint(j.name, spherical_joint)
                assert output.isValidJointIndex(joint_index)
            else:
                joint = to_idyntree_joint(j, links_map[j.parent], links_map[j.child])
                joint_index = output.addJoint(j.name, joint)
                assert output.isValidJointIndex(joint_index)

    frames_list = [f + "_fixed_joint" for f in model.frames]
    for name in model.joints:
        if name in frames_list:
            joint = model.joints[name]
            frame_position = idyntree.bindings.Position.FromPython(
                _to_sequence(joint.origin.xyz)
            )
            frame_transform = idyntree.bindings.Transform()
            frame_transform.setRotation(
                idyntree.bindings.Rotation.RPY(*_to_sequence(joint.origin.rpy))
            )
            frame_transform.setPosition(frame_position)
            frame_name = joint.name.replace("_fixed_joint", "")

            ok = output.addAdditionalFrameToLink(
                joint.parent,
                frame_name,
                frame_transform,
            )
            assert ok

    model_reducer = idyntree.bindings.ModelLoader()
    model_reducer.loadReducedModelFromFullModel(output, model.actuated_joints)
    output_reduced = model_reducer.model().copy()

    assert output_reduced.isValid()
    return output_reduced


class _URDFOrigin:
    """Lightweight stand-in for urdf_parser_py.urdf.Pose."""

    def __init__(self, xyz, rpy):
        self.xyz = xyz
        self.rpy = rpy


class _URDFInertia:
    """Lightweight stand-in for urdf_parser_py.urdf.Inertia."""

    def __init__(self, ixx, ixy, ixz, iyy, iyz, izz):
        self.ixx = ixx
        self.ixy = ixy
        self.ixz = ixz
        self.iyy = iyy
        self.iyz = iyz
        self.izz = izz


class _URDFInertial:
    """Lightweight stand-in for urdf_parser_py.urdf.Inertial."""

    def __init__(self, mass, origin, inertia):
        self.mass = mass
        self.origin = origin
        self.inertia = inertia


class _URDFLimit:
    """Lightweight stand-in for urdf_parser_py.urdf.JointLimit."""

    def __init__(self, lower, upper, effort=0.0, velocity=0.0):
        self.lower = lower
        self.upper = upper
        self.effort = effort
        self.velocity = velocity


class _URDFLinkProxy:
    """Lightweight proxy mimicking urdf_parser_py.urdf.Link for StdLink."""

    def __init__(self, name, inertial, visuals=None, collisions=None):
        self.name = name
        self.inertial = inertial
        self.visuals = visuals or []
        self.collisions = collisions or []


class _URDFJointProxy:
    """Lightweight proxy mimicking urdf_parser_py.urdf.Joint for StdJoint."""

    _IDYNTREE_JOINT_TYPE_MAP = {
        "FixedJoint": "fixed",
        "RevoluteJoint": "revolute",
        "PrismaticJoint": "prismatic",
        "SphericalJoint": "spherical",
    }

    def __init__(self, name, joint_type, parent, child, origin, axis=None, limit=None):
        self.name = name
        self.joint_type = joint_type
        self.parent = parent
        self.child = child
        self.origin = origin
        self.axis = axis
        self.limit = limit


def _idyntree_rotation_to_rpy(
    rotation: idyntree.bindings.Rotation,
) -> list[float]:
    """Extract RPY angles from an iDynTree Rotation."""
    return list(rotation.asRPY().toNumPy())


def _idyntree_position_to_xyz(
    position: idyntree.bindings.Position,
) -> list[float]:
    """Extract xyz from an iDynTree Position."""
    return list(position.toNumPy())


def _extract_link_proxy(
    idyn_model: idyntree.bindings.Model, link_index: int
) -> _URDFLinkProxy:
    """Build a _URDFLinkProxy from an iDynTree link."""
    link_name = idyn_model.getLinkName(link_index)
    idyn_link = idyn_model.getLink(link_index)
    sp_inertia = idyn_link.getInertia()

    mass = sp_inertia.getMass()
    com = _idyntree_position_to_xyz(sp_inertia.getCenterOfMass())

    # getRotationalInertiaWrtCenterOfMass returns the 3x3 inertia at the COM
    rot_inertia = sp_inertia.getRotationalInertiaWrtCenterOfMass()
    ixx = rot_inertia.getVal(0, 0)
    ixy = rot_inertia.getVal(0, 1)
    ixz = rot_inertia.getVal(0, 2)
    iyy = rot_inertia.getVal(1, 1)
    iyz = rot_inertia.getVal(1, 2)
    izz = rot_inertia.getVal(2, 2)

    inertia = _URDFInertia(ixx, ixy, ixz, iyy, iyz, izz)
    origin = _URDFOrigin(xyz=com, rpy=[0.0, 0.0, 0.0])
    inertial = _URDFInertial(mass=mass, origin=origin, inertia=inertia)

    return _URDFLinkProxy(name=link_name, inertial=inertial)


def _infer_idyntree_joint_type(idyn_joint) -> str:
    """Infer the adam joint type string from an iDynTree IJoint.

    SWIG may return the base ``IJoint`` wrapper even for concrete
    subclasses, so ``type().__name__`` is unreliable.  We fall back to
    ``isinstance`` checks and ``getNrOfDOFs()`` when the class name is
    not recognised.
    """
    # Fast path: class name matches a known concrete type
    class_name = type(idyn_joint).__name__
    known = _URDFJointProxy._IDYNTREE_JOINT_TYPE_MAP.get(class_name)
    if known is not None:
        return known

    # Try isinstance checks (works when SWIG preserves the hierarchy)
    for cls_name, type_str in _URDFJointProxy._IDYNTREE_JOINT_TYPE_MAP.items():
        cls = getattr(idyntree.bindings, cls_name, None)
        if cls is not None and isinstance(idyn_joint, cls):
            return type_str

    # Fall back to DOF-based classification
    ndofs = idyn_joint.getNrOfDOFs()
    if ndofs == 0:
        return "fixed"
    if ndofs >= 3:
        return "spherical"
    # 1-DOF: try to distinguish revolute vs prismatic via isinstance
    prism_cls = getattr(idyntree.bindings, "PrismaticJoint", None)
    if prism_cls is not None and isinstance(idyn_joint, prism_cls):
        return "prismatic"
    # Default to revolute for any remaining 1-DOF joint
    return "revolute"


def _extract_joint_proxy(
    idyn_model: idyntree.bindings.Model, joint_index: int
) -> _URDFJointProxy:
    """Build a _URDFJointProxy from an iDynTree joint."""
    joint_name = idyn_model.getJointName(joint_index)
    idyn_joint = idyn_model.getJoint(joint_index)

    joint_type = _infer_idyntree_joint_type(idyn_joint)

    # Parent / child link names
    first_idx = idyn_joint.getFirstAttachedLink()
    second_idx = idyn_joint.getSecondAttachedLink()

    # Determine parent and child using the model's traversal
    traversal = idyntree.bindings.Traversal()
    idyn_model.computeFullTreeTraversal(traversal)

    parent_idx, child_idx = first_idx, second_idx
    for i in range(traversal.getNrOfVisitedLinks()):
        visited = traversal.getLink(i)
        visited_idx = visited.getIndex()
        if visited_idx == first_idx or visited_idx == second_idx:
            # The first one encountered in traversal is the parent
            parent_idx = visited_idx
            child_idx = second_idx if visited_idx == first_idx else first_idx
            break

    parent_name = idyn_model.getLinkName(parent_idx)
    child_name = idyn_model.getLinkName(child_idx)

    # Rest transform: parent -> child
    rest_transform = idyn_joint.getRestTransform(parent_idx, child_idx)
    xyz = _idyntree_position_to_xyz(rest_transform.getPosition())
    rpy = _idyntree_rotation_to_rpy(rest_transform.getRotation())
    origin = _URDFOrigin(xyz=xyz, rpy=rpy)

    axis = None
    limit = None
    if joint_type in ("revolute", "prismatic"):
        # SWIG returns the base IJoint wrapper which lacks getAxis().
        # Downcast to the concrete type to access it.
        concrete = None
        if joint_type == "revolute":
            concrete = idyn_joint.asRevoluteJoint()
        elif joint_type == "prismatic":
            concrete = idyn_joint.asPrismaticJoint()

        if concrete is not None:
            idyn_axis = concrete.getAxis(child_idx)
            direction = idyn_axis.getDirection()
            axis = [direction.getVal(0), direction.getVal(1), direction.getVal(2)]
        else:
            # Last resort: extract from the 6-D motion subspace vector
            msv = idyn_joint.getMotionSubspaceVector(0, child_idx, parent_idx)
            msv_np = msv.toNumPy().flatten()
            axis = [float(msv_np[3]), float(msv_np[4]), float(msv_np[5])]

        if idyn_joint.hasPosLimits():
            lower = idyn_joint.getMinPosLimit(0)
            upper = idyn_joint.getMaxPosLimit(0)
            limit = _URDFLimit(lower=lower, upper=upper)

    return _URDFJointProxy(
        name=joint_name,
        joint_type=joint_type,
        parent=parent_name,
        child=child_name,
        origin=origin,
        axis=axis,
        limit=limit,
    )


def from_idyntree_model(
    idyn_model: idyntree.bindings.Model,
    joints_name_list: list[str] | None = None,
    math: SpatialMath | None = None,
) -> Model:
    """Convert an iDynTree Model to an adam Model.

    Args:
        idyn_model: the iDynTree model to convert.
        joints_name_list: the list of actuated joint names.  If ``None``,
            all non-fixed joints present in the iDynTree model are used.
        math: the SpatialMath backend to use.  Defaults to the NumPy backend.

    Returns:
        Model: the adam model.
    """
    if math is None:
        from adam.numpy.numpy_like import SpatialMath as NumpySpatialMath

        math = NumpySpatialMath()

    # -- Build links ----------------------------------------------------------
    links: list[StdLink] = []
    for link_idx in range(idyn_model.getNrOfLinks()):
        proxy = _extract_link_proxy(idyn_model, link_idx)
        links.append(StdLink(proxy, math))

    # -- Build joints ---------------------------------------------------------
    joints: list[StdJoint] = []
    for joint_idx in range(idyn_model.getNrOfJoints()):
        proxy = _extract_joint_proxy(idyn_model, joint_idx)
        joints.append(StdJoint(proxy, math))

    # -- Build frames (additional frames in iDynTree) -------------------------
    frames: list[StdLink] = []
    for frame_idx in range(idyn_model.getNrOfFrames()):
        # iDynTree frame indices start after link indices; skip link frames
        if idyn_model.isValidLinkIndex(frame_idx):
            continue
        frame_name = idyn_model.getFrameName(frame_idx)
        # Skip frames that correspond to links (link-frames share the index)
        if any(l.name == frame_name for l in links):
            continue
        # Create a zero-inertia link proxy for the frame
        frame_proxy = _URDFLinkProxy(name=frame_name, inertial=None)
        frames.append(StdLink(frame_proxy, math))
        # Also create a fixed joint connecting the frame to its parent link
        parent_link_idx = idyn_model.getFrameLink(frame_idx)
        parent_link_name = idyn_model.getLinkName(parent_link_idx)
        frame_transform = idyn_model.getFrameTransform(frame_idx)
        frame_xyz = _idyntree_position_to_xyz(frame_transform.getPosition())
        frame_rpy = _idyntree_rotation_to_rpy(frame_transform.getRotation())
        frame_joint_proxy = _URDFJointProxy(
            name=frame_name + "_fixed_joint",
            joint_type="fixed",
            parent=parent_link_name,
            child=frame_name,
            origin=_URDFOrigin(xyz=frame_xyz, rpy=frame_rpy),
        )
        joints.append(StdJoint(frame_joint_proxy, math))

    # -- Determine actuated joints --------------------------------------------
    if joints_name_list is None:
        joints_name_list = [j.name for j in joints if j.type != "fixed"]

    # -- Assemble the Model using factory-compatible path ---------------------
    from adam.model.tree import Tree

    # Assign DOF indices to actuated joints
    current_pos_idx = 0
    for joint_name in joints_name_list:
        for j in joints:
            if j.name != joint_name:
                continue
            dofs = getattr(j, "dofs", 1)
            if j.type == "fixed" or dofs == 0:
                j.idx = None
            elif dofs == 1:
                j.idx = current_pos_idx
            else:
                j.idx = tuple(range(current_pos_idx, current_pos_idx + dofs))
            current_pos_idx += dofs
            break

    tree = Tree.build_tree(links=links, joints=joints)

    joints_dict: dict[str, Joint] = {j.name: j for j in joints}
    links_dict: dict[str, Link] = {l.name: l for l in links}
    frames_dict: dict[str, Link] = {f.name: f for f in frames}

    return Model(
        name=idyn_model.toString(),
        links=links_dict,
        frames=frames_dict,
        joints=joints_dict,
        tree=tree,
        NDoF=current_pos_idx,
        actuated_joints=joints_name_list,
    )
