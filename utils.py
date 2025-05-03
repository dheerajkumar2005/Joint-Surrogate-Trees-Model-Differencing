import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict

from rule import Rule

or_nodes = 0

def simpler_common_trunk_tree_to_digraph(root: dict):
    """
    visualisation utility to prepare a networkx graph and draw it using graphviz.
    returns a networkx digraph from the common trunk tree dictionary (jst).
    Args:
        root: joint surrogate tree structure (a dictionary)

    Returns:
        T: a networkx digraph representation of the same jst
    """
    T = nx.DiGraph()
    color1 = "lightpink"  # node color for tree1
    color2 = "lightsalmon"  # node color for tree2

    leaf_colors = ["antiquewhite", "lightcyan", "grey70", "antiquewhite3", "aquamarine",
                   "floralwhite"]  # assume max 6 classes

    def _recurse(root, parentid, direction, style="", fillcolor="lightgrey"):
        vnum = T.number_of_nodes()

        is_diverging = 'tree1' in root

        if is_diverging:
            # separate trees
            global or_nodes
            # print(or_nodes)
            label = f"Vo[{or_nodes}]"
            T.add_node(vnum, shape="circle", label=f"vo{or_nodes}", style="dotted")
            or_nodes += 1

            _recurse(root["tree1"], vnum, '1', style="filled", fillcolor=color1)
            _recurse(root["tree2"], vnum, '2', style="filled", fillcolor=color2)

        else:
            # common part
            if 'cutoff' in root:
                # split node
                # label = 'X[{}] < {}'.format(root['index_col'], np.around(root['cutoff'], 2))
                label = '{}({}) < {}'.format(root['col'], root['index_col'], np.around(root['cutoff'], 2))
                # print(label)
                T.add_node(vnum, label=label, style=style, fillcolor=fillcolor)

                if 'left' in root:
                    _recurse(root['left'], vnum, 'T', style, fillcolor)
                    _recurse(root['right'], vnum, 'F', style, fillcolor)

            else:
                # leaf node
                label = 'label={}\n#samples={}'.format(root['val'], root['dist'])
                if root['ispure']:
                    s = "pure"
                else:
                    s = "impure"
                # label = '{}\n{}'.format(root['val'], s)
                label = '{}'.format(s)
                T.add_node(vnum, label=label, shape="box", style='filled', fillcolor=leaf_colors[root['val']])

        if parentid is not None:
            if direction in ['1', '2']:
                # this is where the trees diverge
                T.add_edge(parentid, vnum, label=direction, style='dashed')
            else:
                # regular left-right edge
                T.add_edge(parentid, vnum, label=direction)

    _recurse(root, None, None)
    # re-assign global or node counting variable to zero before returning
    global or_nodes
    or_nodes = 0
    return T


def graph_to_jpg(T, path='abcd.jpg'):
    """
    save the networkx digraph form of JST to an image.
    Args:
        T: networkx digraph object of the jst as returned from `simpler_common_trunk_tree_to_digraph` function
        path: path to save the jpg file.

    Returns:
        path of the saved image
    """
    from networkx.drawing.nx_agraph import to_agraph
    A = to_agraph(T)
    A.layout('dot')
    A.draw(path, format="jpg")
    return path

def visualize_jst(joint_surrogate_tree: dict, path='jst.jpg'):
    T = simpler_common_trunk_tree_to_digraph(joint_surrogate_tree)
    return graph_to_jpg(T, path)

def load_bc_dataset():
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer(as_frame=True)
    return data["data"], data["target"]

def load_iris_dataset():
    from sklearn.datasets import load_iris
    data = load_iris(as_frame=True)
    return data["data"], data["target"]