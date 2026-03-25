import urdf_parser_py.urdf

from adam.core.spatial_math import SpatialMath
from adam.model import Joint, Link, ModelFactory, StdJoint, StdLink
from adam.model.std_factories.std_model import get_xml_string, urdf_remove_sensors_tags
from adam.model.std_factories.std_model import URDFModelFactory
from adam.model.std_factories.urdf_visual import normalize_urdf_visual
from adam.parametric.model import ParametricJoint, ParametricLink


class URDFParametricModelFactory(ModelFactory):
    """This factory generates robot elements from urdf_parser_py parametrized w.r.t. link lengths and densities

    Args:
        ModelFactory: the Model factory
    """

    # TODO: path can be either a path and an urdf-string, leaving path for back compatibility, to be changed to meaningfull name
    def __init__(
        self,
        path: str,
        math: SpatialMath,
        links_name_list: list,
        length_multiplier,
        densities,
    ):
        self.math = math
        self.resource_roots = URDFModelFactory._infer_resource_roots(path)
        xml_string = get_xml_string(path)
        self.links_name_list = links_name_list

        # Read URDF, but before passing it to urdf_parser_py get rid of all sensor tags
        # sensor tags are valid elements of URDF (see ),
        # but they are ignored by urdf_parser_py, that complains every time it sees one.
        # As there is nothing to be fixed in the used models, and it is not useful
        # to have a useless and noisy warning, let's remove before hands all the sensor elements,
        # that anyhow are not parser by urdf_parser_py or adam
        # See https://github.com/ami-iit/ADAM/issues/59
        xml_string_without_sensors_tags = urdf_remove_sensors_tags(xml_string)

        self.urdf_desc = urdf_parser_py.urdf.URDF.from_xml_string(
            xml_string_without_sensors_tags
        )
        self.name = self.urdf_desc.name
        self.length_multiplier = length_multiplier
        self.densities = densities

    def get_joints(self) -> list[Joint]:
        """
        Returns:
            list[Joint]: build the list of the joints
        """
        return [self.build_joint(j) for j in self.urdf_desc.joints]

    def get_links(self) -> list[Link]:
        """
        Returns:
            list[Link]: build the list of the links
        """
        return [
            self.build_link(l) for l in self.urdf_desc.links if l.inertial is not None
        ]

    def get_frames(self) -> list[StdLink]:
        """
        Returns:
            list[Link]: build the list of the links
        """
        return [self.build_link(l) for l in self.urdf_desc.links if l.inertial is None]

    def build_joint(self, joint: urdf_parser_py.urdf.Joint) -> Joint:
        """
        Args:
            joint (Joint): the urdf_parser_py joint

        Returns:
            StdJoint/ParametricJoint: our joint representation
        """
        if joint.parent in self.links_name_list:
            index_link = self.links_name_list.index(joint.parent)
            link_parent = self.get_element_by_name(joint.parent)
            parent_link_parametric = ParametricLink(
                link_parent,
                self.math,
                self.length_multiplier[index_link],
                self.densities[index_link],
            )
            return ParametricJoint(joint, self.math, parent_link_parametric)

        return StdJoint(joint, self.math)

    def build_link(self, link: urdf_parser_py.urdf.Link) -> Link:
        """
        Args:
            link (Link): the urdf_parser_py link

        Returns:
            StdLink/ParametricLink: our link representation
        """
        if link.name in self.links_name_list:
            index_link = self.links_name_list.index(link.name)
            return ParametricLink(
                link,
                self.math,
                self.length_multiplier[index_link],
                self.densities[index_link],
            )
        visuals = [
            normalize_urdf_visual(
                visual, math=self.math, resource_roots=self.resource_roots
            )
            for visual in link.visuals
        ]
        return StdLink(link, self.math, visuals=visuals)

    def get_element_by_name(self, link_name):
        """
        Args:
            link_name (Link): the link name

        Returns:
            Link: the urdf parser link object associated to the link name
        """
        link_list = [
            corresponding_link
            for corresponding_link in self.urdf_desc.links
            if corresponding_link.name == link_name
        ]
        if len(link_list) != 0:
            return link_list[0]
        else:
            return None
