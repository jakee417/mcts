import networkx as nx
from pyvis.network import Network


from src.node import Node


def convert_root_to_networkx(root: Node) -> nx.Graph:
    G = nx.DiGraph()
    queue = [(root, "0")]
    while queue:
        node, node_id = queue.pop(0)
        tokens = "\n".join([f"{i + 1}: {token}" for i, token in enumerate(node.tokens)])
        G.add_node(
            node_id,
            size=20,
            label=f"{node.visits}",
            title=f"""Visits: {node.visits}, Prob: {round(node.prob, 3)}, Value: {round(node.value, 3)}
Tokens:\n{tokens}
""",
            group=node.visits,
        )
        queue.extend(
            [(child, node_id + "_" + str(i)) for i, child in enumerate(node.children)]
        )
        G.add_edges_from(
            [(node_id, node_id + "_" + str(i)) for i, _ in enumerate(node.children)]
        )
    assert nx.is_directed_acyclic_graph(G)
    return G


def create_graph_html(root: Node, filename: str, height: str):
    G = convert_root_to_networkx(root=root)
    net = Network(
        height=height,
        width="100%",
        directed=True,
        neighborhood_highlight=True,
        cdn_resources="remote",
        bgcolor="#F2FFFFFF",
        layout=True,
    )
    net.from_nx(G)
    net.show(filename, notebook=False)
