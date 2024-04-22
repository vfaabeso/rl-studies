# the graph module
class Graph:
    def __init__(self):
        self.node_reference = {}
        self.edges = {}

    def add_node(self, node: Node) -> None:
        # add to the node reference
        self.node_reference[node.name] = node
        if node.name not in self.node_reference.keys():
            self.edges[node.name] = []

    def add_edge(self, node_A: str, node_B: str) -> None:
        self.edges[node_A].append(node_B)
        self.edges[node_B].append(node_A)

    def get(self, node_name: str) -> Node:
        return self.node_reference[node_name]

class Node:
    def __init__(self, name: str, info: dict):
        self.name = name
        self.info = info