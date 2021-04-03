import GPR
import pandas as pd
import numpy as np
from rdkit import Chem
import copy
import collections


def get_PC_of_surrounding_atoms(mol, pc, atomIdx, verbose=False):
    """Get the partial charges of the surrounding atoms:
        1 bond away and 2 bonds away
    Args:
        mol(rdkit.mol) - moleculue
        pc(pd.DataFrame) - partial charges for each atom
        atomIdx(int) - atom center
    """

    result = []
    mol_info = GPR.RD.get_mol_info(mol)
    # atomIdx one bond away indices
    a1ba = [i for i,Idx in enumerate(list(mol_info["neighbors Idx"])) if atomIdx in Idx]
    if verbose: print(f"Indices one bond away: {a1ba}")
    # atomIdx two bond away indices
    a2ba = [i for i,Idx in enumerate(list(mol_info["neighbors Idx"]))
            for atomIdx_1ba in a1ba if atomIdx_1ba in Idx]
    a2ba.remove(atomIdx)
    if verbose: print(f"Indcies two bond away: {a2ba}")
    result.append(
            {"partial charge of aoi": pc["partial charge"].iloc[atomIdx].mean(),
            "avg partial charge 1ba": pc["partial charge"].iloc[a1ba].mean(),
                "avg partial charge 2ba": pc["partial charge"].iloc[a2ba].mean()}
            )
    df = pd.DataFrame(result)
    if verbose: print(df)
    return df


def split_transitions(df, microtransitions):
    x,y = microtransitions.transpose()
    molx = Chem.AddHs(Chem.MolFromSmiles(df["smiles"][x[0]]))
    moly = Chem.AddHs(Chem.MolFromSmiles(df["smiles"][y[0]]))
    X = GPR.RD.get_mol_info(molx)
    Y = GPR.RD.get_mol_info(moly)
    nHx = int(X["nHydrogens"].sum())
    nHy = int(Y["nHydrogens"].sum())
    if nHx > nHy: return (x,y)
    elif nHx < nHy: return (y,x)


def get_atom_of_interest(prot_mol_info, deprot_mol_info):
    """Finding the atom of interest (aoi) when there could be different sets of
    atom indices.
    Args:
        prot_mol_info(pd.DataFrame) - GPR.RD.mol_info
        deprot_mol_info(pd.DataFrame) - GPR.RD.mol_info
    Returns:
        protAOI,deprotAOI(int,int)
    NOTE: This is one of the ugliest functions I've ever written. In addition,
    it is only ~98% accurate. It is very close to being 100%, but i'm not going
    to touch it at this point...
    """

    pd.options.display.max_rows = 50
    protAOI = None
    FPs = ["basic","acceptor","donor","acidic"]
    #print(prot_mol_info)
    #print(deprot_mol_info)
    #prot_list = list(filter(("H").__ne__, prot_mol_info["symbol"]))
    #deprot_list = list(filter(("H").__ne__, deprot_mol_info["symbol"]))
    #if prot_list == deprot_list:
    for k,row1 in enumerate(prot_mol_info.iterrows()):
        prot_index = row1[0]
        prot_symbol = row1[1]["symbol"]
        prot_charge = row1[1]["formal charge"]
        prot_nHs = row1[1]["nHydrogens"]
        prot_neighbors = row1[1]["neighbors"]
        if prot_nHs >= 1: attachedH = True
        else: attachedH = False
        if "H" in prot_neighbors: prot_neighbors.remove("H")
        for j,row2 in enumerate(deprot_mol_info.iterrows()):
            deprot_index = row2[0]
            deprot_symbol = row2[1]["symbol"]
            deprot_charge = row2[1]["formal charge"]
            deprot_nHs = row2[1]["nHydrogens"]
            deprot_neighbors = row2[1]["neighbors"]

            prot_fp_cond = set(FPs).intersection(
                    set(prot_mol_info["Morgan fingerprints"][prot_index]))
            deprot_fp_cond = set(FPs).intersection(
                    set(deprot_mol_info["Morgan fingerprints"][deprot_index]))

            if (prot_fp_cond or deprot_fp_cond):
                if prot_symbol == deprot_symbol:
                    if prot_nHs != deprot_nHs:
                        if len(sorted(prot_neighbors)) == len(sorted(deprot_neighbors)):
                            if collections.Counter(prot_neighbors) == collections.Counter(deprot_neighbors):
                                if attachedH:
                                    protAOI = prot_index
                                    deprotAOI = deprot_index
                                    break
            else: continue
        if protAOI != None: break
    if protAOI == None:
        for k,row1 in enumerate(prot_mol_info.iterrows()):
            prot_index = row1[0]
            prot_symbol = row1[1]["symbol"]
            prot_charge = row1[1]["formal charge"]
            prot_nHs = row1[1]["nHydrogens"]
            prot_neighbors = row1[1]["neighbors"]
            if prot_nHs >= 1: attachedH = True
            else: attachedH = False
            if "H" in prot_neighbors: prot_neighbors.remove("H")
            for j,row2 in enumerate(deprot_mol_info.iterrows()):
                deprot_index = row2[0]
                deprot_symbol = row2[1]["symbol"]
                deprot_charge = row2[1]["formal charge"]
                deprot_nHs = row2[1]["nHydrogens"]
                deprot_neighbors = row2[1]["neighbors"]

                prot_fp_cond = set(FPs).intersection(
                        set(prot_mol_info["Morgan fingerprints"][prot_index]))
                deprot_fp_cond = set(FPs).intersection(
                        set(deprot_mol_info["Morgan fingerprints"][deprot_index]))

                if (prot_fp_cond or deprot_fp_cond):
                    if prot_symbol == deprot_symbol:
                        if prot_nHs == deprot_nHs:
                            if prot_charge >= deprot_charge:
                                prot_neighbors = list(filter(("H").__ne__, prot_neighbors))
                                deprot_neighbors = list(filter(("H").__ne__, deprot_neighbors))
                                if len(sorted(prot_neighbors)) == len(sorted(deprot_neighbors)):
                                    if collections.Counter(prot_neighbors) == collections.Counter(deprot_neighbors):
                                        if attachedH:
                                            protAOI = prot_index
                                            deprotAOI = deprot_index
                                            break
                else: continue
            if protAOI != None: break

    print(prot_mol_info.iloc[[protAOI]])
    print(deprot_mol_info.iloc[[deprotAOI]])
    return int(protAOI), int(deprotAOI)


## Hold get_atom_of_interest:{{{
#def get_atom_of_interest(prot_mol_info, deprot_mol_info):
#    """Finding the atom of interest (aoi) when there could be different sets of
#    atom indices.
#    Args:
#        prot_mol_info(pd.DataFrame) - RD.GPR.mol_info
#        deprot_mol_info(pd.DataFrame) - RD.GPR.mol_info
#    Returns:
#        protAOI,deprotAOI(int,int)
#    """
#
#    pd.options.display.max_rows = 50
#    print(prot_mol_info)
#    print(deprot_mol_info)
#    protAOI = None
#    prot_list = list(filter(("H").__ne__, prot_mol_info["symbol"]))
#    deprot_list = list(filter(("H").__ne__, deprot_mol_info["symbol"]))
#    if prot_list == deprot_list:
#        # NOTE: If atoms have the same index
#        for k,row1 in enumerate(prot_mol_info.iterrows()):
#            prot_index = row1[0]
#            prot_symbol = row1[1]["symbol"]
#            prot_charge = row1[1]["formal charge"]
#            prot_nHs = row1[1]["nHydrogens"]
#            prot_neighbors = row1[1]["neighbors"]
#            prot_fp_cond = set(["basic","acceptor","donor"]).intersection(
#                    set(prot_mol_info["Morgan fingerprints"][prot_index]))
#            if (prot_fp_cond):
#                if "H" in prot_neighbors:
#                    protAOI = prot_index
#                    deprotAOI = protAOI
#                    break
#                else: continue
#            if protAOI != None: break
#    else:
#        # NOTE: If atoms don't have the same index
#        for k,row1 in enumerate(prot_mol_info.iterrows()):
#            prot_index = row1[0]
#            prot_symbol = row1[1]["symbol"]
#            prot_charge = row1[1]["formal charge"]
#            prot_nHs = row1[1]["nHydrogens"]
#            prot_neighbors = row1[1]["neighbors"]
#            if "H" in prot_neighbors: prot_neighbors.remove("H")
#            for j,row2 in enumerate(deprot_mol_info.iterrows()):
#                deprot_index = row2[0]
#                deprot_symbol = row2[1]["symbol"]
#                deprot_charge = row2[1]["formal charge"]
#                deprot_nHs = row2[1]["nHydrogens"]
#                deprot_neighbors = row2[1]["neighbors"]
#
#                prot_fp_cond = set(["basic","acceptor","donor"]).intersection(
#                        set(prot_mol_info["Morgan fingerprints"][prot_index]))
#                deprot_fp_cond = set(["basic","acceptor","donor"]).intersection(
#                        set(deprot_mol_info["Morgan fingerprints"][deprot_index]))
#
#                if (prot_fp_cond or deprot_fp_cond):
#                    if prot_symbol == deprot_symbol:
#                        if prot_nHs != deprot_nHs:
#                            if len(sorted(prot_neighbors)) == len(sorted(deprot_neighbors)):
#                                if collections.Counter(prot_neighbors) == collections.Counter(deprot_neighbors):
#                                    protAOI = prot_index
#                                    deprotAOI = deprot_index
#                                    break
#                else: continue
#            if protAOI != None: break
#    print(protAOI)
#    print(deprotAOI)
#    print(prot_mol_info.iloc[[protAOI]])
#    print(deprot_mol_info.iloc[[deprotAOI]])
#    return int(protAOI), int(deprotAOI)
##:}}}
#




def get_features(df, microtransitions, verbose=False):
    """Construct feature vector of physical characteristics

    Feature vector
    ==============
    - [x] difference in enthalpy                (1)
    - [x] partial bond order                    (1)
    - [x] partial charges                       (6)
    - [x] difference in solvation free energy   (1)
    - [x] SASA of the deprotonated atom         (1)

    """

    protonated, deprotonated = split_transitions(df, microtransitions)
    transitions, features = [], []
    for i in range(len(deprotonated)):
        # Get deprotonated and protonated molecules for each transition
        protMolIdx, deprotMolIdx = protonated[i], deprotonated[i]
        protName = f'{df["macrostate ID"][protMolIdx]}_{df["microstate ID"][protMolIdx]}'
        deprotName = f'{df["macrostate ID"][deprotMolIdx]}_{df["microstate ID"][deprotMolIdx]}'
        if verbose: print(f'Transition: {deprotName} -> {protName}...')
        #df["smiles"][protMolIdx], df["smiles"][deprotMolIdx] = GPR.OE.set_stereochemistry(df["smiles"][protMolIdx]), GPR.OE.set_stereochemistry(df["smiles"][deprotMolIdx])
        protSmiles, deprotSmiles = df["smiles"][protMolIdx], df["smiles"][deprotMolIdx]
        transitions.append({
            "deprotonated microstate ID": deprotName, "protonated microstate ID": protName,
            "deprotonated microstate smiles": deprotSmiles, "protonated microstate smiles": protSmiles})
        protMol, deprotMol = Chem.MolFromSmiles(protSmiles, sanitize=True), Chem.MolFromSmiles(deprotSmiles, sanitize=True)
        #protMol, deprotMol = Chem.MolFromSmiles(Chem.MolToSmiles(protMol)), Chem.MolFromSmiles(Chem.MolToSmiles(deprotMol))
        #mol_info, deprot_mol_info = GPR.RD.get_mol_info(protMol), GPR.RD.get_mol_info(deprotMol)

        # Get the minimized structures
        #protFilename, deprotFilename = protSmiles+".pdb", deprotSmiles+".pdb"
        protFilename, deprotFilename = "prot.mol", "deprot.mol"

#        # Minimization with Psi4:{{{
#        print("Minimizing...")
#        protMinMol = GPR.minimize_structure(protSmiles)
#        # FIXME: You should only need to minimize once!!!!
#        # You are wasting time minimizing twice because the molecule will be the
#        #same the difference is only a fucking hydrogen!!!!
#        Chem.rdmolfiles.MolToPDBFile(protMinMol, "protMol.pdb")
#        deprotMinMol = GPR.minimize_structure(deprotSmiles)
#        Chem.rdmolfiles.MolToPDBFile(deprotMinMol, "deprotMol.pdb")
#        print("Done.")
#        protMinMol = GPR.RD.minimize_structure(protMinMol, maxIters=5000, nConfs=100, filename=protFilename)
#        deprotMinMol = GPR.RD.minimize_structure(deprotMinMol, maxIters=5000, nConfs=100, filename=deprotFilename)
#        #:}}}
#
        # Minimization with RDKit
        # NOTE: I'm not sure why a molecule would not mnimize (RDKit) "could not triangle smooth bonds"
        try:
            protMinMol = GPR.RD.minimize_structure(protMol, maxIters=5000, nConfs=100, mmffVariant="MMFF94s", filename=protFilename)
            deprotMinMol = GPR.RD.minimize_structure(deprotMol, maxIters=5000, nConfs=100, mmffVariant="MMFF94s", filename=deprotFilename)
        except(Exception) as e: continue

        prot_mol_info, deprot_mol_info = GPR.RD.get_mol_info(protMinMol), GPR.RD.get_mol_info(deprotMinMol)

        # NOTE: Calculate partial charges for the minimized molecules
        try:
            pc1_prot, pc1_deprot = GPR.RD.compute_AM1BCC_charges(protMinMol), GPR.RD.compute_AM1BCC_charges(deprotMinMol)
        except(Exception) as e:
            pc1_prot, pc1_deprot = np.nan, np.nan
        try:
            pc2_prot, pc2_deprot = GPR.RD.compute_Gasteiger_charges(protMinMol), GPR.RD.compute_Gasteiger_charges(deprotMinMol)
        except(Exception) as e:
            pc2_prot, pc2_deprot = np.nan, np.nan
        try:
            pc3_prot, pc3_deprot = GPR.RD.compute_EH(protMinMol), GPR.RD.compute_EH(deprotMinMol)
        except(Exception) as e:
            pc3_prot, pc3_deprot = np.nan, np.nan

        # NOTE: Getting Partial Charge Features
        # get the atom index for the atoms with Morgan Fingerprints we are interested in
        fp_atomIdxs = [row[0] for k,row in enumerate(prot_mol_info.iterrows()) if set(["basic","acceptor","donor"]).intersection(set(prot_mol_info["Morgan fingerprints"][k]))]
        atomIdxs = [row[0] for k,row in enumerate(prot_mol_info.iterrows())]

        # find the atom that is deprotonated
        pd.options.display.max_rows = 999
        #print(prot_mol_info)
        #print(deprot_mol_info)


        #"nHydrogens": sum([H for i,H in enumerate(mol_info["nHydrogens"])
        #    if (set(["basic","acceptor","donor"]).intersection(set(mol_info["Morgan fingerprints"][i])))
        #    or (mol_info["formal charge"][i] >= 1)]),
        #print(fp_atomIdxs)
        try: # FIXME: This needs to work correctly otherwise it's all wrong.
            protAOI, deprotAOI = get_atom_of_interest(prot_mol_info, deprot_mol_info)
        except(Exception) as e: continue

        #print("\n")
        #print(protAOI)
        #print(deprotAOI)

        #NOTE: if using fingerprint atom indices
        #protAtomIndices = [Idx for Idx in fp_atomIdxs if list(prot_mol_info["nHydrogens"])[Idx] != list(deprot_mol_info["nHydrogens"])[Idx]][0]
        #NOTE: if finding the difference in hydrogens of every atom with "H" as a neighbor
        #print(prot_mol_info.iloc[:,2])
        #print(prot_mol_info["H" in prot_mol_info.iloc[:,2]])

        #protAtomIndices = [Idx for Idx in fp_atomIdxs if "H" in list(prot_mol_info["neighbors"])[Idx]][0]
        #print(protAtomIndices)

        #exit()
        #protAtomIdx = int([Idx for Idx in fp_atomIdxs if list(prot_mol_info["nHydrogens"])[Idx] != list(deprot_mol_info["nHydrogens"])[Idx]][0])
        #deprotAtomIdx = int([Idx for Idx in fp_atomIdxs if list(deprot_mol_info["nHydrogens"])[Idx] != list(deprot_mol_info["nHydrogens"])[Idx]][0])
        #print(protAtomIdx)


        # NOTE: Getting Change in Free Energy of Solvation
        if verbose: print(f'charge states: {int(df["charge state"][deprotMolIdx])} and {int(df["charge state"][protMolIdx])}')
        #prot_mol2, deprot_mol2 = protFilename.replace(".sdf",".mol2"), deprotFilename.replace(".sdf",".mol2")
        #GPR.toolbox.mol2_from_PDB(protFilename, filename=protFilename)
        #GPR.toolbox.mol2_from_PDB(deprotFilename, filename=deprotFilename)
        protSolvEnergy = GPR.OE.get_freeE_solv(protFilename, int(df["charge state"][protMolIdx]))
        deprotSolvEnergy = GPR.OE.get_freeE_solv(deprotFilename, int(df["charge state"][deprotMolIdx]))
        dGsolv = protSolvEnergy - deprotSolvEnergy

        # NOTE: Getting Solvent Accessible Surface Area for deprotonated atom
        #GPR.toolbox.PDB_from_MOL(deprotFilename, deprotPDB)
        deprotPDB = deprotFilename.replace(".mol",".pdb")
        #sasa = GPR.RD.get_SASA(deprotPDB, atom_indices=[deprotAOI])[0]
        sasa_shrake = GPR.RD.calc_SASA(deprotMinMol, atom_indices=[deprotAOI], alg="Shrake")[0]
        sasa_lee = GPR.RD.calc_SASA(deprotMinMol, atom_indices=[deprotAOI], alg="Lee")[0]

        # NOTE: Getting Mayer Partial Bond Order of bond between proton and ionizable group
        bo = GPR.RD.set_overlap_populations(protMinMol)
        #bo = GPR.RD.get_bond_order(protMinMol)
        #bo = GPR.OE.get_Mayer_partial_bond_order(protFilename.replace(".mol","_ensemble.pdb"), verbose=False)
        #print(bo)
        try:
            pd.options.display.max_rows = 50
            atomIndices = np.where((bo["begin atom Idx"]==int(protAOI)) & (bo["end atom symbol"]=="H"))[0]
            #print(atomIndices)
            bond_order = float(bo.iloc[atomIndices]["Mulliken population"].mean())
            #print(bo.iloc[atomIndices]["Mulliken population"])
        except(TypeError): # shoot out an error if more than one proton attached to atom
            bond_order = 0
            if verbose: print("No Hydrogen, so bond order = 0")

        # NOTE: Getting Change in Enthalpy
        try:
            prot_S,prot_G = GPR.OE.get_thermodynamics(protFilename, verbose=False)
            deprot_S,deprot_G = GPR.OE.get_thermodynamics(deprotFilename, verbose=False)
        except(Exception) as e: continue

        dH = ((prot_G-deprot_G)+298.*(prot_S-deprot_S))/1000.
        #if verbose: print(f"∆G = %6.4f kJ/mol"%dGsolv)
        if verbose: print(f"∆G = %6.4f kJ/mol"%dGsolv)
        if verbose: print(f"∆H = %6.4f kJ/mol"%dH)
        #exit()
        #prot_H = GPR.OE.get_entropy(protSmiles)
        #deprot_H = GPR.OE.get_entropy(deprotSmiles)
        nan = {"partial charge of aoi": np.nan,
                "avg partial charge 1ba": np.nan,
                "avg partial charge 2ba": np.nan}

        try: prot_pc1 = get_PC_of_surrounding_atoms(protMol, pc1_prot, protAOI)
        except(Exception) as e: prot_pc1 = nan
        try: prot_pc2 = get_PC_of_surrounding_atoms(protMol, pc2_prot, protAOI)
        except(Exception) as e: prot_pc2 = nan
        try: prot_pc3 = get_PC_of_surrounding_atoms(protMol, pc3_prot, protAOI)
        except(Exception) as e: prot_pc3 = nan
        try: deprot_pc1 = get_PC_of_surrounding_atoms(deprotMol, pc1_deprot, deprotAOI)
        except(Exception) as e: deprot_pc1 = nan
        try: deprot_pc2 = get_PC_of_surrounding_atoms(deprotMol, pc2_deprot, deprotAOI)
        except(Exception) as e: deprot_pc2 = nan
        try: deprot_pc3 = get_PC_of_surrounding_atoms(deprotMol, pc3_deprot, deprotAOI)
        except(Exception) as e: deprot_pc3 = nan

        features.append({
            "AM1BCC partial charge (prot. atom)":                prot_pc1["partial charge of aoi"],
            "AM1BCC partial charge (deprot. atom)":              deprot_pc1["partial charge of aoi"],
            "AM1BCC partial charge (prot. atoms 1 bond away)":   prot_pc1["avg partial charge 1ba"],
            "AM1BCC partial charge (deprot. atoms 1 bond away)": deprot_pc1["avg partial charge 1ba"],
            "AM1BCC partial charge (prot. atoms 2 bond away)":   prot_pc1["avg partial charge 2ba"],
            "AM1BCC partial charge (deprot. atoms 2 bond away)": deprot_pc1["avg partial charge 2ba"],

            "Gasteiger partial charge (prot. atom)":                prot_pc2["partial charge of aoi"],
            "Gasteiger partial charge (deprot. atom)":              deprot_pc2["partial charge of aoi"],
            "Gasteiger partial charge (prot. atoms 1 bond away)":   prot_pc2["avg partial charge 1ba"],
            "Gasteiger partial charge (deprot. atoms 1 bond away)": deprot_pc2["avg partial charge 1ba"],
            "Gasteiger partial charge (prot. atoms 2 bond away)":   prot_pc2["avg partial charge 2ba"],
            "Gasteiger partial charge (deprot. atoms 2 bond away)": deprot_pc2["avg partial charge 2ba"],

            "Extented Hückel partial charge (prot. atom)":                prot_pc3["partial charge of aoi"],
            "Extented Hückel partial charge (deprot. atom)":              deprot_pc3["partial charge of aoi"],
            "Extented Hückel partial charge (prot. atoms 1 bond away)":   prot_pc3["avg partial charge 1ba"],
            "Extented Hückel partial charge (deprot. atoms 1 bond away)": deprot_pc3["avg partial charge 1ba"],
            "Extented Hückel partial charge (prot. atoms 2 bond away)":   prot_pc3["avg partial charge 2ba"],
            "Extented Hückel partial charge (deprot. atoms 2 bond away)": deprot_pc3["avg partial charge 2ba"],

            "∆G_solv (kJ/mol) (prot-deprot)": dGsolv,
            "SASA (Shrake)": sasa_shrake,
            "SASA (Lee)": sasa_lee,
            "Bond Order": bond_order,
            "Change in Enthalpy (kJ/mol) (prot-deprot)": dH
            })

    transitions = pd.DataFrame(transitions)
    feature_vector = pd.DataFrame(features)
    total_df = pd.concat([transitions, feature_vector], axis=1)
    return total_df




#        lenPC = len(list(pc[protName])) # use modulus to wrap around list
#        features.append({
#            "AM1BCC partial charge (prot. atom)": pc1[protName][atomIdx],
#            "AM1BCC partial charge (deprot. atom)": pc1[deprotName][atomIdx],
#            "AM1BCC partial charge (prot. atoms 1 bond away)": np.average([list(pc1[protName])[(atomIdx-1) % lenPC], list(pc1[protName])[(atomIdx+1) % lenPC]]),
#            "AM1BCC partial charge (deprot. atoms 1 bond away)": np.average([list(pc1[deprotName])[(atomIdx-1) % lenPC], list(pc1[deprotName])[(atomIdx+1) % lenPC]]),
#            "AM1BCC partial charge (prot. atoms 2 bond away)": np.average([list(pc1[protName])[(atomIdx-2) % lenPC], list(pc1[protName])[(atomIdx+2) % lenPC]]),
#            "AM1BCC partial charge (deprot. atoms 2 bond away)": np.average([list(pc1[deprotName])[(atomIdx-2) % lenPC], list(pc1[deprotName])[(atomIdx+2) % lenPC]]),
#
#            "Gasteiger partial charge (prot. atom)": pc2[protName][atomIdx],
#            "Gasteiger partial charge (deprot. atom)": pc2[deprotName][atomIdx],
#            "Gasteiger partial charge (prot. atoms 1 bond away)": np.average([list(pc2[protName])[(atomIdx-1) % lenPC], list(pc2[protName])[(atomIdx+1) % lenPC]]),
#            "Gasteiger partial charge (deprot. atoms 1 bond away)": np.average([list(pc2[deprotName])[(atomIdx-1) % lenPC], list(pc2[deprotName])[(atomIdx+1) % lenPC]]),
#            "Gasteiger partial charge (prot. atoms 2 bond away)": np.average([list(pc2[protName])[(atomIdx-2) % lenPC], list(pc2[protName])[(atomIdx+2) % lenPC]]),
#            "Gasteiger partial charge (deprot. atoms 2 bond away)": np.average([list(pc2[deprotName])[(atomIdx-2) % lenPC], list(pc2[deprotName])[(atomIdx+2) % lenPC]]),
#
#            "Extented Hückel partial charge (prot. atom)": pc3[protName][atomIdx],
#            "Extented Hückel partial charge (deprot. atom)": pc3[deprotName][atomIdx],
#            "Extented Hückel partial charge (prot. atoms 1 bond away)": np.average([list(pc3[protName])[(atomIdx-1) % lenPC], list(pc3[protName])[(atomIdx+1) % lenPC]]),
#            "Extented Hückel partial charge (deprot. atoms 1 bond away)": np.average([list(pc3[deprotName])[(atomIdx-1) % lenPC], list(pc3[deprotName])[(atomIdx+1) % lenPC]]),
#            "Extented Hückel partial charge (prot. atoms 2 bond away)": np.average([list(pc3[protName])[(atomIdx-2) % lenPC], list(pc3[protName])[(atomIdx+2) % lenPC]]),
#            "Extented Hückel partial charge (deprot. atoms 2 bond away)": np.average([list(pc3[deprotName])[(atomIdx-2) % lenPC], list(pc3[deprotName])[(atomIdx+2) % lenPC]]),
#
#            "∆G_solv (kJ/mol) (prot-deprot)": dGsolv,
#            "SASA": sasa,
#            "Bond Order": bond_order,
#            "Change in Enthalpy (kJ/mol) (prot-deprot)": dH
#            })



#Hold:{{{
#def get_features(df, microtransitions, verbose=False):
#    """Construct feature vector of physical characteristics
#
#    Feature vector
#    ==============
#    - [x] difference in enthalpy                (1)
#    - [x] partial bond order                    (1)
#    - [x] partial charges                       (6)
#    - [x] difference in solvation free energy   (1)
#    - [x] SASA of the deprotonated atom         (1)
#
#    """
#
#    protonated, deprotonated = split_transitions(df, microtransitions)
#    transitions, features = [], []
#    for i in range(len(deprotonated)):
#        try:
#            # Get deprotonated and protonated molecules for each transition
#            protMolIdx, deprotMolIdx = protonated[i], deprotonated[i]
#            protName = f'{df["macrostate ID"][protMolIdx]}_{df["microstate ID"][protMolIdx]}'
#            deprotName = f'{df["macrostate ID"][deprotMolIdx]}_{df["microstate ID"][deprotMolIdx]}'
#            if verbose: print(f'Transition: {deprotName} -> {protName}...')
#            #df["smiles"][protMolIdx], df["smiles"][deprotMolIdx] = GPR.OE.set_stereochemistry(df["smiles"][protMolIdx]), GPR.OE.set_stereochemistry(df["smiles"][deprotMolIdx])
#            protSmiles, deprotSmiles = df["smiles"][protMolIdx], df["smiles"][deprotMolIdx]
#            transitions.append({
#                "deprotonated microstate ID": deprotName, "protonated microstate ID": protName,
#                "deprotonated microstate smiles": deprotSmiles, "protonated microstate smiles": protSmiles})
#            protMol, deprotMol = Chem.MolFromSmiles(protSmiles, sanitize=True), Chem.MolFromSmiles(deprotSmiles, sanitize=True)
#            #protMol, deprotMol = Chem.MolFromSmiles(Chem.MolToSmiles(protMol)), Chem.MolFromSmiles(Chem.MolToSmiles(deprotMol))
#            #prot_mol_info, deprot_mol_info = GPR.RD.get_mol_info(protMol), GPR.RD.get_mol_info(deprotMol)
#
#            # Get the minimized structures
#            #protFilename, deprotFilename = protSmiles+".pdb", deprotSmiles+".pdb"
#            protFilename, deprotFilename = "prot.mol", "deprot.mol"
#
##            # Minimization with Psi4:{{{
##            print("Minimizing...")
##            protMinMol = GPR.minimize_structure(protSmiles)
##            # FIXME: You should only need to minimize once!!!!
##            # You are wasting time minimizing twice because the molecule will be the
##            #same the difference is only a fucking hydrogen!!!!
##            Chem.rdmolfiles.MolToPDBFile(protMinMol, "protMol.pdb")
##            deprotMinMol = GPR.minimize_structure(deprotSmiles)
##            Chem.rdmolfiles.MolToPDBFile(deprotMinMol, "deprotMol.pdb")
##            print("Done.")
##            protMinMol = GPR.RD.minimize_structure(protMinMol, maxIters=5000, nConfs=100, filename=protFilename)
##            deprotMinMol = GPR.RD.minimize_structure(deprotMinMol, maxIters=5000, nConfs=100, filename=deprotFilename)
##            #:}}}
##
#            # Minimization with RDKit
#            protMinMol = GPR.RD.minimize_structure(protMol, maxIters=5000, nConfs=100, mmffVariant="MMFF94s", filename=protFilename)
#            deprotMinMol = GPR.RD.minimize_structure(deprotMol, maxIters=5000, nConfs=100, mmffVariant="MMFF94s", filename=deprotFilename)
#            prot_mol_info, deprot_mol_info = GPR.RD.get_mol_info(protMinMol), GPR.RD.get_mol_info(deprotMinMol)
#
#            # NOTE: Calculate partial charges for the minimized molecules
#            pc1_prot, pc1_deprot = GPR.RD.compute_AM1BCC_charges(protMinMol), GPR.RD.compute_AM1BCC_charges(deprotMinMol)
#            pc2_prot, pc2_deprot = GPR.RD.compute_Gasteiger_charges(protMinMol), GPR.RD.compute_Gasteiger_charges(deprotMinMol)
#            pc3_prot, pc3_deprot = GPR.RD.compute_EH(protMinMol), GPR.RD.compute_EH(deprotMinMol)
#
#            # NOTE: Getting Partial Charge Features
#            # get the atom index for the atoms with Morgan Fingerprints we are interested in
#            fp_atomIdxs = [row[0] for k,row in enumerate(prot_mol_info.iterrows()) if set(["basic","acceptor","donor"]).intersection(set(prot_mol_info["Morgan fingerprints"][k]))]
#            atomIdxs = [row[0] for k,row in enumerate(prot_mol_info.iterrows())]
#
#            # find the atom that is deprotonated
#            try:
#                atomIdx = [Idx for Idx in fp_atomIdxs if prot_mol_info["nHydrogens"][Idx] != deprot_mol_info["nHydrogens"][Idx]][0]
#            except(IndexError, KeyError) as e:
#                #pd.options.display.max_rows = 999
#                #print(prot_mol_info)
#                #print(deprot_mol_info)
#                atomIdx = int([Idx for Idx in atomIdxs if prot_mol_info["nHydrogens"][Idx] != deprot_mol_info["nHydrogens"][Idx]][0])
#
#            # NOTE: Getting Change in Free Energy of Solvation
#            if verbose: print(f'charge states: {int(df["charge state"][deprotMolIdx])} and {int(df["charge state"][protMolIdx])}')
#            #prot_mol2, deprot_mol2 = protFilename.replace(".sdf",".mol2"), deprotFilename.replace(".sdf",".mol2")
#            #GPR.toolbox.mol2_from_PDB(protFilename, filename=protFilename)
#            #GPR.toolbox.mol2_from_PDB(deprotFilename, filename=deprotFilename)
#            protSolvEnergy = GPR.OE.get_freeE_solv(protFilename, int(df["charge state"][protMolIdx]))
#            deprotSolvEnergy = GPR.OE.get_freeE_solv(deprotFilename, int(df["charge state"][deprotMolIdx]))
#            dGsolv = protSolvEnergy - deprotSolvEnergy
#            if verbose: print(f"∆G = %6.4f kJ/mol"%dGsolv)
#
#            # NOTE: Getting Solvent Accessible Surface Area for deprotonated atom
#            #GPR.toolbox.PDB_from_MOL(deprotFilename, deprotPDB)
#            deprotPDB = deprotFilename.replace(".mol",".pdb")
#            #sasa = GPR.RD.get_SASA(deprotPDB, atom_indices=[atomIdx])[0]
#            sasa_shrake = GPR.RD.calc_SASA(deprotMinMol, atom_indices=[atomIdx], alg="Shrake")[0]
#            sasa_lee = GPR.RD.calc_SASA(deprotMinMol, atom_indices=[atomIdx], alg="Lee")[0]
#
#            # NOTE: Getting Mayer Partial Bond Order of bond between proton and ionizable group
#            bo = GPR.RD.set_overlap_populations(protMinMol)
#            #bo = GPR.RD.get_bond_order(protMinMol)
#            #bo = GPR.OE.get_Mayer_partial_bond_order(protFilename.replace(".mol","_ensemble.pdb"), verbose=False)
#            try:
#                pd.options.display.max_rows = 50
#                print(atomIdx)
#                atomIndices = np.where((bo["begin atom Idx"]==int(atomIdx)) & (bo["end atom symbol"]=="H"))
#                print(atomIndices)
#                bond_order = float(bo[atomIndices]["Mulliken population"].mean())
#                print(bond_order)
#                exit()
#            except(TypeError): # shoot out an error if more than one proton attached to atom
#                bond_order = 0
#                if verbose: print("No Hydrogen, so bond order = 0")
#            # NOTE: Getting Change in Enthalpy
#            prot_S,prot_G = GPR.OE.get_thermodynamics(protFilename, verbose=False)
#            deprot_S,deprot_G = GPR.OE.get_thermodynamics(deprotFilename, verbose=False)
#            dH = ((prot_G-deprot_G)+298.*(prot_S-deprot_S))/1000.
#            #if verbose: print(f"∆G = %6.4f kJ/mol"%dGsolv)
#            print(f"∆G = %6.4f kJ/mol"%dGsolv)
#            print(f"∆H = %6.4f kJ/mol"%dH)
#            #exit()
#            #prot_H = GPR.OE.get_entropy(protSmiles)
#            #deprot_H = GPR.OE.get_entropy(deprotSmiles)
#
#            prot_pc1 = get_PC_of_surrounding_atoms(protMol, pc1_prot, atomIdx)
#            prot_pc2 = get_PC_of_surrounding_atoms(protMol, pc2_prot, atomIdx)
#            prot_pc3 = get_PC_of_surrounding_atoms(protMol, pc3_prot, atomIdx)
#            deprot_pc1 = get_PC_of_surrounding_atoms(deprotMol, pc1_deprot, atomIdx)
#            deprot_pc2 = get_PC_of_surrounding_atoms(deprotMol, pc2_deprot, atomIdx)
#            deprot_pc3 = get_PC_of_surrounding_atoms(deprotMol, pc3_deprot, atomIdx)
#
#            features.append({
#                "AM1BCC partial charge (prot. atom)":                float(prot_pc1["partial charge of aoi"]),
#                "AM1BCC partial charge (deprot. atom)":              float(deprot_pc1["partial charge of aoi"]),
#                "AM1BCC partial charge (prot. atoms 1 bond away)":   float(prot_pc1["avg partial charge 1ba"]),
#                "AM1BCC partial charge (deprot. atoms 1 bond away)": float(deprot_pc1["avg partial charge 1ba"]),
#                "AM1BCC partial charge (prot. atoms 2 bond away)":   float(prot_pc1["avg partial charge 2ba"]),
#                "AM1BCC partial charge (deprot. atoms 2 bond away)": float(deprot_pc1["avg partial charge 2ba"]),
#
#                "Gasteiger partial charge (prot. atom)":                float(prot_pc2["partial charge of aoi"]),
#                "Gasteiger partial charge (deprot. atom)":              float(deprot_pc2["partial charge of aoi"]),
#                "Gasteiger partial charge (prot. atoms 1 bond away)":   float(prot_pc2["avg partial charge 1ba"]),
#                "Gasteiger partial charge (deprot. atoms 1 bond away)": float(deprot_pc2["avg partial charge 1ba"]),
#                "Gasteiger partial charge (prot. atoms 2 bond away)":   float(prot_pc2["avg partial charge 2ba"]),
#                "Gasteiger partial charge (deprot. atoms 2 bond away)": float(deprot_pc2["avg partial charge 2ba"]),
#
#                "Extented Hückel partial charge (prot. atom)":                float(prot_pc3["partial charge of aoi"]),
#                "Extented Hückel partial charge (deprot. atom)":              float(deprot_pc3["partial charge of aoi"]),
#                "Extented Hückel partial charge (prot. atoms 1 bond away)":   float(prot_pc3["avg partial charge 1ba"]),
#                "Extented Hückel partial charge (deprot. atoms 1 bond away)": float(deprot_pc3["avg partial charge 1ba"]),
#                "Extented Hückel partial charge (prot. atoms 2 bond away)":   float(prot_pc3["avg partial charge 2ba"]),
#                "Extented Hückel partial charge (deprot. atoms 2 bond away)": float(deprot_pc3["avg partial charge 2ba"]),
#
#                "∆G_solv (kJ/mol) (prot-deprot)": dGsolv,
#                "SASA (Shrake)": sasa_shrake,
#                "SASA (Lee)": sasa_lee,
#                "Bond Order": bond_order,
#                "Change in Enthalpy (kJ/mol) (prot-deprot)": dH
#                })
#        except (Exception) as e: print(e)
#
#    transitions = pd.DataFrame(transitions)
#    feature_vector = pd.DataFrame(features)
#    total_df = pd.concat([transitions, feature_vector], axis=1)
#    return total_df
#
#
#:}}}

## def get_atom_of_interest(prot_mol_info, deprot_mol_info):{{{{{{
#
#def get_atom_of_interest(prot_mol_info, deprot_mol_info):
#    """Finding the atom of interest (aoi) when there could be different sets of
#    atom indices.
#    """
#
#    pd.options.display.max_rows = 50
#    #print(prot_mol_info)
#    protAtomIndices = [row[0] for k,row in enumerate(prot_mol_info.iterrows())
#            if set(["H"]).intersection(set(list(prot_mol_info["neighbors"])[k]))]
#    prot_fp_atomIdxs = [row[0] for k,row in enumerate(prot_mol_info.iterrows())
#            if set(["basic","acceptor","donor"]).intersection(set(prot_mol_info["Morgan fingerprints"][k]))]
#
#    protAOI = list(set(protAtomIndices) & set(prot_fp_atomIdxs))
#    protAOI = copy.deepcopy(prot_mol_info.iloc[protAOI])
#    prot_formal_charge = protAOI["formal charge"].mean()
#    neighbors = copy.deepcopy(list(protAOI["neighbors"])[0])
#    #neighbors = list(filter(("H").__ne__, neighbors))
#    deprotAOI = [row[0] for k,row in enumerate(deprot_mol_info.iterrows())
#            if all(neighbor in neighbors for neighbor in list(deprot_mol_info["neighbors"])[k])
#            if (list(protAOI["symbol"])[0] == list(deprot_mol_info["symbol"])[k])]
#
#    #if (list(protAOI["symbol"])[0] == list(deprot_mol_info["symbol"])[k])
#    #if (list(deprot_mol_info["formal charge"])[row[0]] != protAOI["formal charge"].sum())
#
#
##    print(protAOI){{{
##    if len(list(protAOI["symbol"])) > 1:
##        Index = 0
##        for k,row in enumerate(protAOI.iterrows()):
##            mirror = list(row[1]["neighbors"])
##            mirror.remove("H")
##            if (row[1]["symbol"] == list(deprot_mol_info.iloc[deprotAOI]["symbol"])[0]):
##                if (mirror == list(deprot_mol_info.iloc[deprotAOI]["neighbors"])[0]):
##                    Index = int(row[0])
##                    break
##        protAOI = prot_mol_info.loc[[Index]]
## }}}
#
#    protAOI = list(protAOI.index)
#    if (len(protAOI) > 1) and (len(deprotAOI) == 1):
#       # prot_mol_info.iloc[protAOI]
#        for i in list(protAOI):
#            row = prot_mol_info.iloc[i]
#            mirror = list(row["neighbors"])
#            mirror.remove("H")
#            row = prot_mol_info.iloc[i]
#            print(row["symbol"])
#            print(deprotAOI)
#            print(list(deprot_mol_info["symbol"])[deprotAOI[0]])
#            if row["symbol"] == list(deprot_mol_info["symbol"])[deprotAOI[0]]:
#                if (mirror == list(deprot_mol_info.iloc[deprotAOI]["neighbors"])[0]):
#                    protAOI = i
#                    break
#                elif (row["formal charge"] != deprot_mol_info.iloc[deprotAOI]["formal charge"].sum()):
#                    protAOI = i
#                    break
#    #if len(list(deprot_mol_info.iloc[deprotAOI]["formal charge"])) > 1:
#    elif (len(protAOI) == 1) and (len(deprotAOI) > 1):
#        for i in list(deprotAOI):
#            row = deprot_mol_info.iloc[i]
#            #print(row["formal charge"])
#            #print( protAOI["formal charge"])
#            if (row["formal charge"] != prot_mol_info.iloc[protAOI]["formal charge"].sum()):
#                deprotAOI = i
#                break
#
#    #print(deprotAOI)
#
##    neighbors = copy.deepcopy(list(protAOI["neighbors"])[0])
##    prot_nHydrogens = protAOI["nHydrogens"]
##    neighbors = copy.deepcopy(list(filter(("H").__ne__, neighbors)))
##    deprot_mol_info.iloc[deprotAOI] = [row[0] for k,row in enumerate(deprot_mol_info.iterrows())
##            if (list(protAOI["symbol"])[0] == list(deprot_mol_info["symbol"])[k])
##            if ((prot_nHydrogens-1) == list(deprot_mol_info["nHydrogens"])[k])
##            if all(neighbor in list(deprot_mol_info["neighbors"])[k] for neighbor in neighbors)]
#
#    print(prot_mol_info.iloc[protAOI])
#    print(deprot_mol_info.iloc[deprotAOI])
#    if type(protAOI) == list:
#        if len(protAOI) == 1:
#            protAOI = protAOI[0]
#    if type(deprotAOI) == list:
#        if len(deprotAOI) == 1:
#            deprotAOI = deprotAOI[0]
#    return int(protAOI), int(deprotAOI)
#
#
#
#    #atomIdxs = [row[0] for k,row in enumerate(prot_mol_info.iterrows())]
#    #NOTE: if using fingerprint atom indices
#    #protAtomIndices = [Idx for Idx in fp_atomIdxs if list(mol_info["nHydrogens"])[Idx] != list(mol_info["nHydrogens"])[Idx]][0]
#    #NOTE: if finding the difference in hydrogens of every atom with "H" as a neighbor
#    #print(mol_info.iloc[:,2])
#    #print(mol_info["H" in mol_info.iloc[:,2]])
#
## }}}
#






