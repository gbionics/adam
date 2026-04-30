import dataclasses
from typing import Iterable, Iterator, Union

import numpy.typing as npt

from adam.model.abc_factories import Joint, Link


class ReversedJoint(Joint):
    """Reverses the direction of a joint for kinematic-tree re-rooting.

    When re-rooting from ``new_root`` to the original root, each joint on
    the path needs to carry the child from its *new* parent to its *new*
    child (which are swapped with respect to the original).

    Mathematical convention (same as in ``RBDAlgorithms``):
    * ``spatial_transform(q)``  = ``adjoint_inverse(H(q))``
    * ``homogeneous(q)``        = H(q)   (4x4 homogeneous transform parent→child)

    For the reversed joint (new parent ← original child, new child ← original parent):
    * ``H_rev(q)``     = ``H_orig(q)⁻¹``
    * ``X_rev(q)``     = ``adjoint_inverse(H_rev(q))``
                       = ``adjoint_inverse(H_orig(q)⁻¹)``
                       = ``adjoint(H_orig(q))``
    * ``S_rev``        = ``-adjoint(H_orig(0)) @ S_orig``   (constant, pre-computed)
    """

    def __init__(self, original: Joint) -> None:
        self.math = original.math
        self.name = original.name
        self.parent = original.child  # swapped
        self.child = original.parent  # swapped
        self.type = original.type
        self.axis = original.axis
        self.origin = original.origin
        self.limit = original.limit
        self.idx = original.idx
        self.original = original
        # H(0) = H_from_Pos_RPY(origin.xyz, origin.rpy) for all joint types
        # (revolute/prismatic offset and fixed transform coincide at q=0).
        H0 = original.math.H_from_Pos_RPY(original.origin.xyz, original.origin.rpy)
        S_orig = original.motion_subspace()
        adj_H0 = original.math.adjoint(H0)
        self._motion_subspace = -original.math.mtimes(adj_H0, S_orig)

    def homogeneous(self, q: npt.ArrayLike) -> npt.ArrayLike:
        return self.math.homogeneous_inverse(self.original.homogeneous(q))

    def spatial_transform(self, q: npt.ArrayLike) -> npt.ArrayLike:
        return self.math.adjoint(self.original.homogeneous(q))

    def motion_subspace(self) -> npt.ArrayLike:
        return self._motion_subspace


@dataclasses.dataclass
class Node:
    """The node class"""

    name: str
    link: Link
    arcs: list[Joint]
    children: list["Node"]
    parent: Union[Link, None] = None
    parent_arc: Union[Joint, None] = None

    def __hash__(self) -> int:
        return hash(self.name)

    def get_elements(self) -> tuple[Link, Joint, Link]:
        """returns the node with its parent arc and parent link

        Returns:
            tuple[Link, Joint, Link]: the node, the parent_arc, the parent_link
        """
        return self.link, self.parent_arc, self.parent


@dataclasses.dataclass
class Tree(Iterable):
    """The directed tree class"""

    graph: dict[str, Node]
    root: str

    def __post_init__(self):
        self.ordered_nodes_list = self.get_ordered_nodes_list(self.root)

    @staticmethod
    def build_tree(links: list[Link], joints: list[Joint]) -> "Tree":
        """builds the tree from the connectivity of the elements

        Args:
            links (list[Link])
            joints (list[Joint])

        Returns:
            Tree: the directed tree
        """
        nodes: dict[str, Node] = {
            l.name: Node(
                name=l.name, link=l, arcs=[], children=[], parent=None, parent_arc=None
            )
            for l in links
        }

        for joint in joints:
            # don't add the frames
            if joint.parent not in nodes.keys() or joint.child not in nodes.keys():
                continue

            if joint.parent not in {l.name for l in nodes[joint.parent].children}:
                nodes[joint.parent].children.append(nodes[joint.child])
                nodes[joint.parent].arcs.append(joint)
                nodes[joint.child].parent = nodes[joint.parent].link
                nodes[joint.child].parent_arc = joint

        root_link = [l for l in nodes if nodes[l].parent is None]
        if len(root_link) != 1:
            raise ValueError(
                f"Expected only one root, found {len(root_link)}: {root_link}"
            )
        return Tree(nodes, root_link[0])

    def reroot(self, new_root: str) -> "Tree":
        """Return a new ``Tree`` rooted at ``new_root``.

        Joints on the path from ``new_root`` to the original root are
        wrapped in :class:`ReversedJoint` so that their parent/child
        direction, spatial transform, and motion subspace are all consistent
        with the new root.  All other joints are unchanged.

        Args:
            new_root (str): name of the link that should become the new root.

        Returns:
            Tree: a new tree with ``new_root`` as root.

        Example:
            Given a tree ``A → B → C → D`` (with ``E`` as a child of ``A``
            and ``F`` as a child of ``D``) and ``A`` as root, rerooting at
            ``C`` reverses the path ``[C, B, A]`` and leaves subtrees
            hanging off the path (e.g. ``D``, ``E``, ``F``) unchanged::

                Original (root=A):       Rerooted (root=C):
                     A                        C
                    / \\                      / \\
                   B   E                    D   B    <- reversed joint
                   |                        |   |
                   C                        F   A    <- reversed joint
                   |                            |
                   D                            E    <- copied as-is
                   |
                   F
        """
        if new_root == self.root:
            return self

        if new_root not in self.graph:
            raise ValueError(
                f"{new_root!r} is not a link in the robot model. "
                f"Available links: {list(self.graph)}"
            )

        # Build the path [new_root, ..., old_root] following parent pointers.
        path: list[str] = []
        current: str = new_root
        while True:
            path.append(current)
            node = self.graph[current]
            if node.parent is None:
                break
            current = node.parent.name

        # path_set = all link names on the reversal path.
        path_set: set[str] = set(path)

        # Create fresh (disconnected) nodes for every link.
        new_nodes: dict[str, Node] = {
            name: Node(
                name=name,
                link=old_node.link,
                arcs=[],
                children=[],
                parent=None,
                parent_arc=None,
            )
            for name, old_node in self.graph.items()
        }

        # Copy all edges that are NOT on the reversal path.
        for name, old_node in self.graph.items():
            for joint, child_node in zip(old_node.arcs, old_node.children):
                if child_node.name in path_set:
                    continue  # these edges are re-wired below
                new_nodes[name].children.append(new_nodes[child_node.name])
                new_nodes[name].arcs.append(joint)
                new_nodes[child_node.name].parent = new_nodes[name].link
                new_nodes[child_node.name].parent_arc = joint

        # Re-wire path edges in reverse direction using _ReversedJoint wrappers.
        for i in range(len(path) - 1):
            # Original edge: parent=path[i+1], child=path[i]
            orig_joint = self.graph[path[i]].parent_arc
            rev_joint = ReversedJoint(orig_joint)
            # New edge: parent=path[i], child=path[i+1]
            new_nodes[path[i]].children.append(new_nodes[path[i + 1]])
            new_nodes[path[i]].arcs.append(rev_joint)
            new_nodes[path[i + 1]].parent = new_nodes[path[i]].link
            new_nodes[path[i + 1]].parent_arc = rev_joint

        return Tree(new_nodes, new_root)

    def print(self, root):
        """prints the tree

        Args:
            root (str): the root of the tree
        """
        import pptree

        pptree.print_tree(self.graph[root])

    def get_ordered_nodes_list(self, start: str) -> list[str]:
        """get the ordered list of the nodes, given the connectivity

        Args:
            start (str): the start node

        Returns:
            list[str]: the ordered list
        """
        ordered_list = [start]
        self.get_children(self.graph[start], ordered_list)
        return ordered_list

    @classmethod
    def get_children(cls, node: Node, list: list):
        """Recursive method that finds children of child of child
        Args:
            node (Node): the analized node
            list (list): the list of the children that needs to be filled
        """
        if node.children is not []:
            for child in node.children:
                list.append(child.name)
                cls.get_children(child, list)

    def get_idx_from_name(self, name: str) -> int:
        """
        Args:
            name (str): node name

        Returns:
            int: the index of the node in the ordered list
        """
        return self.ordered_nodes_list.index(name)

    def get_name_from_idx(self, idx: int) -> str:
        """
        Args:
            idx (int): the index in the ordered list

        Returns:
            str: the corresponding node name
        """
        return self.ordered_nodes_list[idx]

    def get_node_from_name(self, name: str) -> Node:
        """
        Args:
            name (str): the node name

        Returns:
            Node: the node istance
        """
        return self.graph[name]

    def __iter__(self) -> Iterator[Node]:
        """This method allows to iterate on the model
        Returns:
            Node: the node istance

        Yields:
            Iterator[Node]: the list of the nodes
        """
        yield from [self.graph[name] for name in self.ordered_nodes_list]

    def __reversed__(self) -> Iterator[Node]:
        """
        Returns:
            Node

        Yields:
            Iterator[Node]: the reversed nodes list
        """
        yield from reversed(self)

    def __getitem__(self, key) -> Node:
        """get the item at key in the model

        Args:
            key (Union[int, Slice]): _description_

        Returns:
            Node: _description_
        """
        return self.graph[self.ordered_nodes_list[key]]

    def __len__(self) -> int:
        """
        Returns:
            int: the length of the model
        """
        return len(self.ordered_nodes_list)
