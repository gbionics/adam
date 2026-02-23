import dataclasses
import pathlib

from adam.model.abc_factories import Joint, Link, ModelFactory
from adam.model.tree import Tree


@dataclasses.dataclass(frozen=True, slots=True)
class Model:
    """
    Model class. It describes the robot using links and frames and their connectivity"""

    name: str
    links: dict[str, Link]
    frames: dict[str, Link]
    joints: dict[str, Joint]
    tree: Tree
    NDoF: int
    actuated_joints: list[str]

    @property
    def N(self) -> int:
        """

        Returns:
            int: the number of links in the model
        """
        return len(self.links)

    @staticmethod
    def build(factory: ModelFactory, joints_name_list: list[str] = None) -> "Model":
        """generates the model starting from the list of joints and the links-joints factory

        Args:
            factory (ModelFactory): the factory that generates the links and the joints, starting from a description (eg. urdf)
            joints_name_list (list[str]): the list of the actuated joints

        Returns:
            Model: the model describing the robot
        """

        joints_list = factory.get_joints()
        links_list = factory.get_links()
        frames_list = factory.get_frames()

        # if the joints_name_list is None, set it to the list of all the joints that are not fixed
        if joints_name_list is None:
            joints_name_list = [
                joint.name for joint in joints_list if joint.type != "fixed"
            ]
            print(
                f"joints_name_list is None. Setting it to the list of all the joints that are not fixed: {joints_name_list}"
            )

        # check if the joints in the list are in the model
        for joint_str in joints_name_list:
            if joint_str not in [joint.name for joint in joints_list]:
                raise ValueError(
                    f"{joint_str} is not in the robot model. Check the joints_name_list"
                )

        # set idx to the actuated joints
        for [idx, joint_str] in enumerate(joints_name_list):
            for joint in joints_list:
                if joint.name == joint_str:
                    joint.idx = idx

        tree = Tree.build_tree(links=links_list, joints=joints_list)

        # generate some useful dict
        joints: dict[str, Joint] = {joint.name: joint for joint in joints_list}
        links: dict[str, Link] = {link.name: link for link in links_list}
        frames: dict[str, Link] = {frame.name: frame for frame in frames_list}

        return Model(
            name=factory.name,
            links=links,
            frames=frames,
            joints=joints,
            tree=tree,
            NDoF=len(joints_name_list),
            actuated_joints=joints_name_list,
        )

    def get_joints_chain(self, root: str, target: str) -> list[Joint]:
        """generate the joints chains from a link to a link

        Args:
            root (str): the starting link
            target (str): the target link

        Returns:
            list[Joint]: the list of the joints
        """

        if target not in list(self.links) and target not in list(self.frames):
            raise ValueError(f"{target} is not not in the robot model.")

        if target == root:
            return []
        chain = []
        current_node = [
            joint for joint in self.joints.values() if joint.child == target
        ][0]

        chain.insert(0, current_node)
        while current_node.parent != root:
            current_node = [
                joint
                for joint in self.joints.values()
                if joint.child == current_node.parent
            ][0]
            chain.insert(0, current_node)
        return chain

    def get_total_mass(self) -> float:
        """total mass of the robot

        Returns:
            float: the total mass of the robot
        """
        mass = 0.0
        for item in self.links:
            link = self.links[item]
            mass += link.inertial.mass.array
        return mass

    def print_table(self):
        """print the table that describes the connectivity between the elements.
        You need rich to use it
        """
        try:
            from rich.console import Console
            from rich.table import Table
        except Exception:
            print("rich is not installed!")
            return

        console = Console()

        table = Table(show_header=True, header_style="bold red")
        table.add_column("Idx")
        table.add_column("Parent Link")
        table.add_column("Joint name")
        table.add_column("Child Link")
        table.add_column("Type")

        nodes = self.tree.graph

        j = 1
        for item in nodes:
            if len(nodes[item].children) != 0:
                for arc in nodes[item].arcs:
                    table.add_row(str(j), item, arc.name, arc.child, arc.type)
                    j += 1

        console.print(table)

    def to_usd(
        self,
        usd_output_path: str | pathlib.Path,
        *,
        robot_prim_path: str = "/Robot",
    ) -> pathlib.Path:
        """Export this model to a USD file containing one articulation robot."""
        from adam.model.conversions.usd import model_to_usd

        return model_to_usd(
            model=self,
            usd_output_path=usd_output_path,
            robot_prim_path=robot_prim_path,
        )
