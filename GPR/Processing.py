import sys,os
from rdkit import Chem
from rdkit.Chem import PandasTools
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
import uncertainties as u ################ Error Prop. Library
import uncertainties.unumpy as unumpy #### Error Prop.
from uncertainties import umath
from itertools import combinations

def get_species_dictionary(input_file):

    input_file = pd.read_csv(input_file, index_col=4)
    pos_charges = set(list(input_file.index))
    species_dict = {i: list(input_file.iloc[np.where(input_file.index==i)]["microstate ID"]) for i in pos_charges}
    return species_dict

def get_micro_transitions(input_file, column=4, ref=None):
    """Note that this is an exaustive list of all microstate transitions

    Args:
        df(pd.DataFrame) - should be a dataframe with columns "smiles",
            "macrostate ID", "microstate ID"
        ref(int) - index for the molecule in DataFrame to be considered the center of the cycle

    Returns:
        pd.DataFrame - microstate transitions

    """

    df = pd.read_csv(input_file, index_col=False)
    key = list(df.keys())[column]
    transitions = {}
    startIdx = np.where(np.array(list(df[key])) == int(df[key].min()))[0]
    start = int(df.iloc[startIdx][key].min())
    maxH = int(df[key].max())
    indices = np.where(np.int(start+1) == np.array(list(df[key])))
    microstates = list(df.iloc[indices]["microstate ID"])
    for microstate in list(df.iloc[startIdx]["microstate ID"]):
        transitions[str(microstate)] = microstates
    #start += 1
    while start <= maxH:
        for microstate in microstates:
            indices = np.where(np.int(start+1) == np.array(list(df[key])))
            new_microstates = list(df.iloc[indices]["microstate ID"])
            transitions[str(microstate)] = new_microstates
            #transitions.append({str(microstate): new_microstates})
        start += 1
        microstates = new_microstates
    #return pd.DataFrame(transitions, index=[item.keys() for item in transitions])
    if ref == "000":
        # make sure there's transitions from 000 -> 1, 2, 3
        keys = transitions.keys()
        for key in keys:
            if key in ['micro001', 'micro002', 'micro003']:
                if ('micro000' not in transitions[key]) and (key not in transitions['micro000']):
                    if 'micro000' in keys: transitions['micro000'].append(key)
                    else: transitions[key].append('micro000')
    return transitions

def get_micro_transitions_from_pairs(input_file, pairs):
    """Note that this is an exaustive list of all microstate transitions

    Args:
        df(pd.DataFrame) - should be a dataframe with columns "smiles",
            "macrostate ID", "microstate ID"
        ref(int) - index for the molecule in DataFrame to be considered the center of the cycle

    Returns:
        pd.DataFrame - microstate transitions

    """

    df = pd.read_csv(input_file, index_col=False)
    transitions = {pair[0].split("_")[-1]: [] for pair in pairs.to_numpy()}
    for pair in pairs.to_numpy():
        prot, deprot = pair[0].split("_")[-1],pair[1].split("_")[-1]
        transitions[str(prot)].append(deprot)
    return transitions



def get_macroscopic_pKas(deprotonated_charge, input_file, results_file, features_file):
    """Calculate macropKa from micropkas
    Args:
        deprotonated_charge(int) - charge of deprotonated species
        input_file(pd.DataFrame) -
        results_file(pd.DataFrame) - typei pKa dataframe
        features_file(pd.DataFrame) -
    """

    species_dict = get_species_dictionary(input_file)
    ###############################################################
    # FIXME:
    #results = pd.read_pickle(glob.glob(results_file)[0])
    results = pd.read_pickle(results_file)
    results = results.dropna()
    total_feat = pd.read_pickle(features_file)
    results = results[["posterior","posterior_std"]]
    results.columns = ["pKa", "pKa std"]
    results["Protonated"] = [x.split("_")[-1] for x in list(total_feat["protonated microstate ID"])]
    results["Deprotonated"] = [x.split("_")[-1] for x in list(total_feat["deprotonated microstate ID"])]
    #transitions = get_micro_transitions(input_file)
    ###############################################################
    #results["Ka"] = np.power(10.0, -results["pKa"])
    Ka_list,Ka_std_list = [],[]
    pKas = list(results["pKa"])
    pKas_std = list(results["pKa std"])
    for i in range(len(pKas)):
        pKa = u.ufloat(pKas[i], pKas_std[i])
        Ka = umath.pow(10.0, -pKa)
        Ka_list.append(Ka.nominal_value)
        Ka_std_list.append(Ka.std_dev)
    results["Ka"] = Ka_list
    results["Ka std"] = Ka_std_list
    k_macro = 0
    for deprot in species_dict[deprotonated_charge]:
        k_micro_inverted = 0
        for prot in species_dict[deprotonated_charge + 1]:
            indices = np.where((results["Protonated"] == prot) & (results["Deprotonated"] == deprot))
            for row_id, row in results.loc[indices].iterrows():
                k_micro_inverted += 1.0 / u.ufloat(row["Ka"], row["Ka std"])
        # If there were no equilibrium constants for this pair, dont add
        if k_micro_inverted != 0: k_macro += 1.0 / k_micro_inverted
    #if k_macro != 0: return -np.log10(k_macro)
    if k_macro != 0: return -umath.log10(k_macro)
    else:
        #print(results, species_dict[deprotonated_charge], species_dict[deprotonated_charge + 1])
        raise ValueError("Error in a graph occurred.")



def DrawMolZoomed(smiles, legend=None, subImgSize=(100, 100)):
    from PIL import Image
    from io import BytesIO
    from rdkit.Chem import AllChem
    from rdkit.Chem.Draw import rdMolDraw2D
    from rdkit.Chem import Draw
    import matplotlib.pyplot as plt

    mol = mol = Chem.MolFromSmiles(smiles)
    fullSize = (subImgSize[0], subImgSize[1])
    full_image = Image.new('RGBA', fullSize )
    d2d = rdMolDraw2D.MolDraw2DCairo(subImgSize[0], subImgSize[1])
    opts = d2d.drawOptions()
    opts.legendFontSize = 20
    d2d.SetFontSize(1)
    d2d.DrawMolecule(mol, legend)
    d2d.FinishDrawing()
    sub = Image.open(BytesIO(d2d.GetDrawingText()))
    full_image.paste(sub)
    return full_image


def graph_from_df(input_file, transitions_file, column=4, results_file=None,
        molSize=0.15, subImgSize=(100,100), figsize=(10,16),
        edge_label_font_size=16, figname="network.png"):
    """Return a DiGraph from Pandas DataFrame.

    Args:
        figsize(tuple) - (height,width)

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
    transitions = np.load(transitions_file)
    input_file = pd.read_csv(input_file, index_col=False)
    key = list(input_file.keys())[column]
    pos = {}
    Hs = input_file[key]
    for i,q in enumerate(Hs):
        m = np.where(q == Hs)[0]
        if len(m) == 1: matches = [0]
        else: matches = np.linspace(-len(m), len(m), len(m))/len(m)
        for x,k in enumerate(m):
            key = str(list(input_file["microstate ID"])[k].split("micro")[-1])
            pos[key] = (-list(Hs)[k], matches[x])
    from_list, to_list = transitions.transpose()
    from_list = [str(list(input_file["microstate ID"])[i].split("micro")[-1]) for i in from_list]
    to_list = [str(list(input_file["microstate ID"])[i].split("micro")[-1]) for i in to_list]
    convert = lambda txt: int(txt) if txt.isdigit() else txt
    micros = sorted(set(np.concatenate([to_list, from_list])), key=lambda x:[convert(s) for s in re.split("([0-9]+)",x)])
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
            graph.add_edge(node[0],node[1], label=Gs[i])
            labels[(node[0],node[1])] = Gs[i]
    else:
        for i,node in enumerate(zip(from_list, to_list)):
            graph.add_edge(node[0],node[1])

    #NOTE: Get Network of states
    fig, (ax) = plt.subplots(1)
    fig.set_figheight(10);fig.set_figwidth(16)
    nx.draw(graph, pos)#, with_labels=False, arrows=False, ax=ax)#, font_weight='bold')
    if results_file: nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels,
            font_size=edge_label_font_size, label_pos=0.6, arrows=True)

    trans=ax.transData.transform
    trans2=fig.transFigure.inverted().transform
    piesize=molSize # this is the image size
    p2=piesize/2.0
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


def rel_free_energy_from_pka(nHij, pKa):
    """Returns the relative free energy in kcal/mol at reference pH of 0."""

    C = 1.36 # kcal/mol : RTln(10), where R (kcal/(mol K))
    Gij = nHij*C*(-pKa)
    return Gij

def get_Gs(ref, input_file, results_file, features_file):
    """
    """

    species_dict = get_species_dictionary(input_file)
    ###############################################################
    # FIXME:
    #results = pd.read_pickle(glob.glob(results_file)[0])
    results = pd.read_pickle(results_file)
    input_file = pd.read_csv(input_file, index_col=False)
    total_feat = pd.read_pickle(features_file)
    results = results[["posterior","posterior_std"]]
    results.columns = ["pKa", "pKa std"]
    results["Protonated"] = [x.split("_")[-1] for x in list(total_feat["protonated microstate ID"])]
    results["Deprotonated"] = [x.split("_")[-1] for x in list(total_feat["deprotonated microstate ID"])]
    #transitions = get_micro_transitions(input_file)
    ###############################################################
    #results["Ka"] = np.power(10.0, -results["pKa"])
    SM = results_file.split("/")[-2]
    #print(f'Small Molecule: {SM}')
    Ka_list,Ka_std_list = [],[]
    pKas = list(results["pKa"])
    pKas_std = list(results["pKa std"])
    Gs = []
    ref_state_ID = f'{SM}_'+ref
    prot_state_IDs = list(total_feat["protonated microstate ID"])
    deprot_state_IDs = list(total_feat["deprotonated microstate ID"])


    # Find what the reference state is connected to.
    immediate_trans_prot_index = np.where(ref_state_ID == np.array(prot_state_IDs))[0]
    immediate_trans_deprot_index = np.where(ref_state_ID == np.array(deprot_state_IDs))[0]

    # Get transitions from microRef --> microxxx
    for protIdx in immediate_trans_prot_index:
        transition = {}
        prot_state_ID = prot_state_IDs[protIdx]
        deprot_state_ID = deprot_state_IDs[protIdx]
        #print(f"Transition: {prot_state_ID} --> {deprot_state_ID}")
        protRow = input_file.iloc[np.where(input_file["microstate ID"]==prot_state_ID.split("_")[-1])[0]]
        protHs = int(protRow["nHydrogens"].to_numpy())
        deprotRow = input_file.iloc[np.where(input_file["microstate ID"]==deprot_state_ID.split("_")[-1])[0]]
        deprotHs = int(deprotRow["nHydrogens"].to_numpy())
        protCharge = int(protRow["charge state"].to_numpy())
        deprotCharge = int(deprotRow["charge state"].to_numpy())
        transIdx = np.where((prot_state_ID == total_feat["protonated microstate ID"]) & (deprot_state_ID == total_feat["deprotonated microstate ID"]))[0]
        resultsCol = results.iloc[transIdx]
        pKa = u.ufloat(resultsCol["pKa"].to_numpy(), resultsCol["pKa std"].to_numpy())
        diff_in_Hs =  deprotHs - protHs
        G = rel_free_energy_from_pka(diff_in_Hs, pKa)

        transition["∆G"] = G.nominal_value
        transition["∆G std"] = G.std_dev
        #transition["∆G SEM"] = 0 # TODO: SEM
        transition["Ref"] = ref_state_ID
        transition["Microstate ID"] = deprot_state_ID
        transition["Charge"] = deprotCharge
        Gs.append(transition)

    # Get transitions from microxxx --> microRef
    for deprotIdx in immediate_trans_deprot_index:
        transition = {}
        prot_state_ID = prot_state_IDs[deprotIdx]
        deprot_state_ID = deprot_state_IDs[deprotIdx]
        #print(f"Transition: {prot_state_ID} --> {deprot_state_ID}")
        protRow = input_file.iloc[np.where(input_file["microstate ID"]==prot_state_ID.split("_")[-1])[0]]
        protHs = int(protRow["nHydrogens"].to_numpy())
        deprotRow = input_file.iloc[np.where(input_file["microstate ID"]==deprot_state_ID.split("_")[-1])[0]]
        deprotHs = int(deprotRow["nHydrogens"].to_numpy())
        protCharge = int(protRow["charge state"].to_numpy())
        deprotCharge = int(deprotRow["charge state"].to_numpy())
        transIdx = np.where((prot_state_ID == total_feat["protonated microstate ID"]) & (deprot_state_ID == total_feat["deprotonated microstate ID"]))[0]
        resultsCol = results.iloc[transIdx]
        pKa = u.ufloat(resultsCol["pKa"].to_numpy(), resultsCol["pKa std"].to_numpy())
        diff_in_Hs = protHs - deprotHs
        G = rel_free_energy_from_pka(diff_in_Hs, pKa)
        # TODO:
        if G.nominal_value == 0.: pass

        transition["∆G"] = G.nominal_value
        transition["∆G std"] = G.std_dev
        #transition["∆G SEM"] = 0 # TODO: SEM
        transition["Ref"] = ref_state_ID
        transition["Microstate ID"] = prot_state_ID
        transition["Charge"] = protCharge
        Gs.append(transition)
    df = pd.DataFrame(Gs)
    return df


def fix_Gs(free_energies, input_file, results_file, features_file, verbose=False):
    """Find the reference state and use that reference state to guide the free energy
    calculations.
    """

    #print(free_energies)
    species_dict = get_species_dictionary(input_file)
    ###############################################################
    # FIXME:
    #results = pd.read_pickle(glob.glob(results_file)[0])
    results = pd.read_pickle(results_file)
    input_file = pd.read_csv(input_file, index_col=False)
    total_feat = pd.read_pickle(features_file)
    results = results[["posterior","posterior_std"]]
    results.columns = ["pKa", "pKa std"]
    results["Protonated"] = [x.split("_")[-1] for x in list(total_feat["protonated microstate ID"])]
    results["Deprotonated"] = [x.split("_")[-1] for x in list(total_feat["deprotonated microstate ID"])]
    ###############################################################
    SM = results_file.split("/")[-2]
    ref_state_ID = list(free_energies["Ref"])[0]
    if verbose: print(ref_state_ID)
    for i in range(len(list(free_energies.index))):
        row = free_energies.iloc[[i]]
        if abs(row["∆G"].mean()) == 0.0:
            if verbose: print(row)
            #print(row)
            microstate_ID = row["Microstate ID"].to_numpy()[0]
            indices = np.where((microstate_ID == total_feat["protonated microstate ID"]))[0]
            #print(indices)
            resultsCol = results.iloc[indices]
            #print(resultsCol)
            state = "Deprotonated"
            if list(resultsCol[state]) == []: state = "Protonated"
            if list(resultsCol[state]) == []: continue
            if verbose: print(state)
            for k in list(resultsCol[state]):
                if k == ref_state_ID.split("_")[-1]: continue
                else:
                    if k in [ID.split("_")[-1] for ID in list(free_energies["Microstate ID"])]:
                        microID = ref_state_ID.split("_")[0]+"_"+k
                        #print(microID)
                        row = free_energies.iloc[np.where(microID == free_energies["Microstate ID"])[0]]
                        #print(row)
                        deltaG = u.ufloat(row["∆G"].mean(), row["∆G std"].mean())
                        break
            row = resultsCol.iloc[np.where(k == resultsCol[state])]
            #print(row)
            pka,pka_std = row["pKa"].mean(),row["pKa std"].mean()
            pKa = u.ufloat(pka, pka_std)
            #print(pKa)
            prot_state_ID,deprot_state_ID = row["Protonated"].to_numpy()[0],row["Deprotonated"].to_numpy()[0]
            protRow = input_file.iloc[np.where(input_file["microstate ID"]==prot_state_ID)[0]]
            protHs = int(protRow["nHydrogens"].to_numpy())
            deprotRow = input_file.iloc[np.where(input_file["microstate ID"]==deprot_state_ID)[0]]
            deprotHs = int(deprotRow["nHydrogens"].to_numpy())
            nHij = protHs - deprotHs
            G = rel_free_energy_from_pka(nHij, pKa)
            new_deltaG = deltaG+G
            free_energies.iloc[i,0] = new_deltaG.nominal_value
            free_energies.iloc[i,1] = new_deltaG.std_dev
            #print(free_energies)
            break
    return free_energies








