import re
import pandas as pd
import numpy as np
from itertools import combinations
from networkx import DiGraph
from network2tikz import plot
from rdkit import Chem
from rdkit.Chem import PandasTools
import matplotlib.pyplot as plt
import networkx as nx


def graph_from_df(input_file, transitions, figname="network.png"):
    """Return a DiGraph from Pandas DataFrame."""

    from networkx import DiGraph
    pos = {}
    Hs = input_file["nHydrogens"]
    for i in range(max(Hs)+1):
        m = np.where(i == Hs)[0]
        if len(m) == 1: matches = [0]
        else: matches = np.linspace(-len(m), len(m), len(m))/len(m)
        for x,k in enumerate(m):
            key = str(input_file["microstate ID"][k].split("micro")[-1])
            pos[key] = (Hs[k], matches[x])
    from_list, to_list = transitions.transpose()
    from_list = [str(input_file["microstate ID"][i].split("micro")[-1]) for i in from_list]
    to_list = [str(input_file["microstate ID"][i].split("micro")[-1]) for i in to_list]
    convert = lambda txt: int(txt) if txt.isdigit() else txt
    micros = sorted(set(np.concatenate([to_list, from_list])), key=lambda x:[convert(s) for s in re.split("([0-9]+)",x)])
    # Direction of edges of the graph is deprotonated -> protonated state
    graph = DiGraph()
    #graph.add_edges_from(zip(from_list, to_list, properties))
    graph.add_edges_from(zip(from_list, to_list))

    #NOTE: Get Network of states
    fig, (ax) = plt.subplots(1)
    fig.set_figheight(10);fig.set_figwidth(10)
    nx.draw(graph, pos, with_labels=True, arrows=False, ax=ax, font_weight='bold')
    #plt.axis('equal', adjustable='datalim')
    ax.set_aspect(aspect=.9, adjustable='box', anchor='C', share=False)
    fig.savefig(figname, bbox_inches='tight', dpi=100)



def tikz_graph_from_df(input_file, transitions, out="network.tex"):
    """Return a DiGraph from Pandas DataFrame.

    https://github.com/hackl/network2tikz
    https://github.com/hackl/network2tikz/blob/master/examples/ex_networkx.py
    """

    import networkx as nx
    #from networkx import DiGraph
    PandasTools.AddMoleculeColumnToFrame(input_file, smilesCol='smiles', molCol='molecule')
    pos = {}
    Hs = input_file["nHydrogens"]
    for i in range(max(Hs)+1):
        m = np.where(i == Hs)[0]
        if len(m) == 1: matches = [0]
        else: matches = np.linspace(-len(m), len(m), len(m))/len(m)
        for x,k in enumerate(m):
            key = str(input_file["microstate ID"][k].split("micro")[-1])
            pos[key] = (Hs[k], matches[x])
    from_list, to_list = transitions.transpose()
    from_list = [str(input_file["microstate ID"][i].split("micro")[-1]) for i in from_list]
    to_list = [str(input_file["microstate ID"][i].split("micro")[-1]) for i in to_list]
    convert = lambda txt: int(txt) if txt.isdigit() else txt
    micros = sorted(set(np.concatenate([to_list, from_list])), key=lambda x:[convert(s) for s in re.split("([0-9]+)",x)])
    # Direction of edges of the graph is deprotonated -> protonated state
    graph = nx.DiGraph(directed=True)
    for i,name in enumerate([ID.split("micro")[-1] for ID in list(input_file["microstate ID"])]):
        graph.add_node(name, name=name, smiles=list(input_file["smiles"])[i])
    nodes = nx.get_node_attributes(graph, 'name')
    for node in zip(from_list, to_list):
        graph.add_edge(node[0],node[1])

    style = {}
    # TODO: Continue adding the style that you want for the LaTeX Network
    style['edge_directed'] = {edge:True for edge in graph.edges()}
    style['edge_directed'] = {edge:True for edge in graph.edges()}

    # general options
    style['unit'] = 'mm'
    style['layout'] = layout
    style["margin"] = {'top':5,'bottom':8,'left':5,'right':5}
    style["canvas"] = (100,60)
    style['keep_aspect_ratio'] = False

    plot(graph, out, **style)



def DrawMolZoomed(smiles, legend=None, subImgSize=(100, 100)):
    from PIL import Image
    from io import BytesIO
    from rdkit.Chem import AllChem
    from rdkit.Chem.Draw import rdMolDraw2D
    from rdkit.Chem import Draw
    import matplotlib.pyplot as plt

    mol = mol = Chem.MolFromSmiles(smiles)
    size = (subImgSize[0], subImgSize[1])
    full_image = Image.new('RGBA', size)
    d2d = rdMolDraw2D.MolDraw2DCairo(subImgSize[0], subImgSize[1])
    opts = d2d.drawOptions()
    opts.legendFontSize = 20
    d2d.SetFontSize(1)
    d2d.DrawMolecule(mol, legend)
    d2d.FinishDrawing()
    sub = Image.open(BytesIO(d2d.GetDrawingText()))
    full_image.paste(sub)
    return full_image


def graph_from_df_(input_file, transitions_file, column=4, results_file=None,
        molSize=0.15, subImgSize=(100,100), figsize=(10,16),
        edge_label_font_size=16, figname="network.png"):
    """Return a DiGraph from Pandas DataFrame.
    Args:
        input_file(str) - path to file that contains information about the nodes
        transitions_file(str) - path to numpy file with: array(array(2)...array(2))
        node-to-node transitions
        column(int) - the column (0-indexing) from `input_file` to be used for
    """

    import networkx as nx
    import re
    import numpy as np
    from itertools import combinations
    from networkx import DiGraph
    from network2tikz import plot
    from rdkit import Chem
    from rdkit.Chem import PandasTools
    import matplotlib.pyplot as plt
    import networkx as nx

    #from networkx import DiGraph
    # load numpy of transitions from node to node
    transitions = np.load(transitions_file)
    # load csv file of informations regarding each node
    input_file = pd.read_csv(input_file, index_col=False)
    # argument to function is the number of the column you wish to use to organize
    # the nodes in. e.g., Organize nodes by the number of ionizable hydrogens
    key = list(input_file.keys())[column]
    pos = {}  # gather the positions of each node
    Hs = input_file[key]
    for i,q in enumerate(Hs):
        m = np.where(q == Hs)[0]
        if len(m) == 1: matches = [0]
        else: matches = np.linspace(-len(m), len(m), len(m))/len(m)
        for x,k in enumerate(m):
            key = str(list(input_file["microstate ID"])[k].split("micro")[-1])
            pos[key] = (-list(Hs)[k], matches[x])
    # transitions from --> to (lists)
    from_list, to_list = transitions.transpose()
    from_list = [str(list(input_file["microstate ID"])[i].split("micro")[-1]) for i in from_list]
    to_list = [str(list(input_file["microstate ID"])[i].split("micro")[-1]) for i in to_list]
    convert = lambda txt: int(txt) if txt.isdigit() else txt
    micros = sorted(set(np.concatenate([to_list, from_list])), key=lambda x:[convert(s) for s in re.split("([0-9]+)",x)])

    #NOTE: Using DiGraph from networkx
    # Direction of edges of the graph is deprotonated -> protonated state
    graph = nx.DiGraph(directed=True)
    for i,name in enumerate([ID.split("micro")[-1] for ID in list(input_file["microstate ID"])]):
        graph.add_node(name, name=name, smiles=list(input_file["smiles"])[i])
    nodes = nx.get_node_attributes(graph, 'name')

    if results_file:
        results = pd.read_pickle(results_file)
        #Gs = [f'{G:.2f}±{list(results["G std"])[i]:.2f}' for i,G in enumerate(list(results["G"]))]
        Gs = [f'{G:.2f}±{list(results["posterior_std"])[i]:.2f}' for i,G in enumerate(list(results["posterior"]))]
        labels = {}
        for i,node in enumerate(zip(from_list, to_list)):
            graph.add_edge(node[0]*scale,node[1]*scale, label=Gs[i])
            labels[(node[0]*scale,node[1])*scale] = Gs[i]
    else:
        for i,node in enumerate(zip(from_list, to_list)):
            graph.add_edge(node[0],node[1])

    # Put the network inside a matplotlib plot image
    #NOTE: Get Network of states
    fig, (ax) = plt.subplots(1)
    fig.set_figheight(figsize[0]);fig.set_figwidth(figsize[1])
    nx.draw(graph, pos)#, with_labels=False, arrows=False, ax=ax)#, font_weight='bold')
    if results_file: nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels,
            font_size=edge_label_font_size, label_pos=0.6, arrows=True)

    trans=ax.transData.transform
    trans2=fig.transFigure.inverted().transform
    piesize=molSize # this is the image size
    p2=piesize/2.0
    # NOTE: Changing each of the nodes to have an image of the molecule instead of a circle
    for node in graph:
        xx,yy=trans(pos[node]) # figure coordinates
        xa,ya=trans2((xx,yy)) # axes coordinates
        a = plt.axes([xa-p2,ya-p2, piesize, piesize])
        smiles = graph.nodes[node]['smiles']
        ID = graph.nodes[node]['name']
        img = DrawMolZoomed(smiles, subImgSize=subImgSize)#, legend=ID)
        a.imshow(img)
        a.axis('off')
        # TODO: make the title the G value
        a.set_title(graph.nodes[node]['name'], fontweight='bold', size=12)
    fig.savefig(figname, bbox_inches='tight', dpi=100)
    return graph,pos


