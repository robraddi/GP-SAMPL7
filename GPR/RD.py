import os
import numpy as np
import pandas as pd
from . import toolbox
from rdkit import Chem
from rdkit.Chem import PandasTools
from rdkit.Chem import AllChem
from rdkit.Chem import rdEHTTools
from rdkit.Chem import rdFreeSASA
from rdkit.Chem import SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import ChemicalForceFields
from itertools import combinations
import mdtraj as md

from . import OE

morgan_fingerprints = dict(acidic="[$([C,S](=[O,S,P])-[O;H1,H0&-1])]",
     basic="[#7;+,$([N;H2&+0][$([C,a]);!$([C,a](=O))]),$([N;H1&+0]([$([C,a]);!$([C,a](=O))])[$([C,a]);!$([C,a](=O))]),$([N;H0&+0]([C;!$(C(=O))])([C;!$(C(=O))])[C;!$(C(=O))])]",
     acceptor="[$([O,S;H1;v2;!$(*-*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=[O,N,P,S])]),n&H0&+0,$([o,s;+0;!$([o,s]:n);!$([o,s]:c:n)])]",
     donor="[$([N;!H0;v3,v4&+1]),$([O,S;H1;+0]),n&H1&+0]",
     aromatic="[a]",
     halogen="[F,Cl,Br,I]",
     sulfonamide="[$([N;H0]S(=O)(=O))]",
     basicNH1="[$([N;H1&+0]([C;!$(C=*)])[C;!$(C=*)])]",
     basicNH2="[$([N;H2&+0][C;!$(C=*)])]",
    )

def get_Morgan_fingerprints(mol):
    """Popular fingerprint that takes into account the neighborhood of each atom.

    fp.GetNumBits() = 2048
    fp.ToBitString() = 1000010...1001
    """

    matches,subs = [],[]
    atoms = ()
    for key,val in morgan_fingerprints.items():
        sub = Chem.MolFromSmarts(val)
        for atom in mol.GetSubstructMatches(sub):
            atoms += atom
            subs.append({"Morgan fingerprints": key,
                "atomIdx": atom[0]})
    submatches = pd.DataFrame(subs)
    for Idx,atom in enumerate(mol.GetAtoms()):
        temp = []
        for i,index in enumerate(submatches["atomIdx"]):
            if Idx == index:
                temp.append(submatches["Morgan fingerprints"][i])
        matches.append({"symbol": atom.GetSymbol(),
            "Morgan fingerprints": temp})
    df = pd.DataFrame(matches)
    return df

def get_mol_img(mol, filename=None):
    img = Chem.Draw.MolToImage(mol)
    if filename != None: img.save(f"{filename}")
    return img

def enumerate_tautomers(m):
    enumerator = rdMolStandardize.TautomerEnumerator()
    canon = enumerator.Canonicalize(m)
    csmi = Chem.MolToSmiles(canon)
    res = [canon]
    tauts = enumerator.Enumerate(m)
    smis = [Chem.MolToSmiles(x) for x in tauts]
    # NOTE: standardize?
    #smis = [rdMolStandardize.StandardizeSmiles(Chem.MolToSmiles(x)) for x in tauts]
    stpl = sorted((x,y) for x,y in zip(smis,tauts) if x!=csmi)
    res += [y for x,y in stpl]
    return res


def get_fingerprints(df, key, fp_col_name="Fingerprint", astype=object):
    if astype == object:
        df[fp_col_name] = [Chem.RDKFingerprint(Chem.MolFromSmiles(x)) for x in list(df[key])]
    if astype == np.array:
        df[fp_col_name] = [Chem.RDKFingerprint(Chem.MolFromSmiles(x)) for x in list(df[key])]
        df[fp_col_name] = [np.array(list(fp.ToBitString())).astype('int8') for fp in list(df[fp_col_name])]
    if astype == str:
        df[fp_col_name] = [Chem.RDKFingerprint(Chem.MolFromSmiles(x)) for x in list(df[key])]
        df[fp_col_name] = [fp.ToBitString() for fp in list(df[fp_col_name])]
    return df

def get_MolWt_from_smiles(smiles):
    return Chem.rdMolDescriptors.CalcExactMolWt(Chem.MolFromSmiles(smiles))


def get_similarity(df, moi_smiles, fp_col_name="Fingerprint"):
    """Returns Dice similarity for count-based fingerprints
    :math:`sim(v_{i},v{j}) = \frac{2.0*\sum_{b}min(v_{ib},v_{jb})}{\sum_{b}v_{ib}+\sum_{b}v_{jb}}`
    :ref:`https://rdkit.org/docs/source/rdkit.DataStructs.cDataStructs.html`
    """

    moi_fp = Chem.RDKFingerprint(Chem.MolFromSmiles(moi_smiles))
    from rdkit import RDConfig
    from rdkit.Dbase.DbConnection import DbConnect
    from rdkit import DataStructs
    #dbName = RDConfig.RDTestDatabase
    #conn = DbConnect(dbName,'simple_mols1')

    metric=DataStructs.DiceSimilarity
    similarity = []
    for fp in list(df[fp_col_name]):
        similarity.append(DataStructs.FingerprintSimilarity(fp, moi_fp, metric))
    df["Fingerprint Similarity"] = similarity
    return df


def save_MolFile(mol, filename):
    Chem.rdmolfiles.MolToMolFile(mol, filename)



#adjustHs
#Kekulize
#setConjugation

def validate_rdkit_mol(mol):
    """
    Sanitizes an RDKit molecules and returns True if the molecule is chemically
    valid.
    :param mol: an RDKit molecule
    :return: True if the molecule is chemically valid, False otherwise
    """

    if len(Chem.GetMolFrags(mol)) > 1:
        return False
    try:
        Chem.SanitizeMol(mol)
        return True
    except ValueError:
        return False


def addH_to_anion(mol, charge=-1, to=0):
    mp = AllChem.MMFFGetMoleculeProperties(mol)
    print("No. of atoms before fix = {:n}\n".format(mol.GetNumAtoms()))
    # Print out all partial charges
    t=0.0
    for i in range(mol.GetNumAtoms()):
        q = mp.GetMMFFPartialCharge(i)
        t += q
        #print(q)
    print("Total Charge before fix = {}\n".format(t))
    # Fix the molecule - first fix the formal charge
    atomIdx = 0
    for i in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(i)
        if atom.GetFormalCharge()== charge:
            print("formal charge changed for atom {}\n".format(i))
            atom.SetFormalCharge(to)
            atomIdx = i
    # double check that the formal charge has been changed.
    for i in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(i)
        if atom.GetFormalCharge()== charge:
            print("ERROR: Atom {:n} Not Changed\n".format(i))
            exit()
    # Fix the molecule - now add hydrogens
#    m2 = Chem.AddHs(mol, addCoords=True)

    RdkitMol_r1 = Chem.RWMol(mol)
    a1 = mol.GetAtomWithIdx(atomIdx)
    a1.SetNoImplicit(True)
    h1 = Chem.Atom(int(1)) # add Hydrogen
    h1.SetNoImplicit(True)
    smi1 = Chem.MolToSmiles(RdkitMol_r1)
    m2 = Chem.MolFromSmiles(smi1)

    mp2 = AllChem.MMFFGetMoleculeProperties(m2)
    print("No. of atoms after fix = {:n}\n".format(m2.GetNumAtoms()))
    # Print out all partial charges
    t=0.0
    for i in range(m2.GetNumAtoms()):
        q = mp2.GetMMFFPartialCharge(i)
        t += q
        #print(q)
    print("Total Charge after fix = {}\n".format(t))
    return m2



def modify_charge(mol, atomIndex, action="addH", charge=-1, to=0, verbose=False):
    """Alter the charge of an atom and and a hydrogen.
    Args:
        mol(object) -
        atomIndex(object) -
        charge(int) -
        to(int) -
    """

    #enumerator = rdMolStandardize.TautomerEnumerator()
    #canon = enumerator.Canonicalize(mol)
    mp = AllChem.MMFFGetMoleculeProperties(mol)
    if verbose: print("No. of atoms before fix = {:n}\n".format(mol.GetNumAtoms()))
    # Print out all partial charges
    t=0.0
    for i in range(mol.GetNumAtoms()):
        q = mp.GetMMFFPartialCharge(i)
        t += q
        #print(q)
    if verbose: print("Total Charge before fix = {}\n".format(t))
    # Fix the molecule - first fix the formal charge
    atom = mol.GetAtomWithIdx(atomIndex)
    if atom.GetFormalCharge() == charge:
        if verbose: print("formal charge changed for atom {}\n".format(atomIndex))
        atom.SetFormalCharge(to)
    # double check that the formal charge has been changed.
    atom = mol.GetAtomWithIdx(atomIndex)
    if atom.GetFormalCharge() == charge:
        if verbose: print("WARNING: Atom {} Not Changed\n".format(atomIndex))

    RdkitMol_r1 = Chem.RWMol(mol)
    # Fix the molecule - now add hydrogens
    if action == "addH":
        a1 = mol.GetAtomWithIdx(atomIndex)
        #RdkitMol_r1 = Chem.AddHs(RdkitMol_r1, onlyOnAtoms=(atomIndex))
        a1.SetNoImplicit(True)
        h1 = Chem.Atom(int(1)) # add Hydrogen
        h1.SetNoImplicit(True)
    try:
        RdkitMol_r1.UpdatePropertyCache(strict=True)
    except(Exception) as e:
        if verbose: print(e)
        RdkitMol_r1.UpdatePropertyCache(strict=False)

    #if action == "removeH":
    #    #NOTE: This code block is required to remove a Hydrogen
    #    a1 = mol.GetAtomWithIdx(atomIndex)
    #    explicitHs = a1.GetNumExplicitHs()
    #    if explicitHs != 0:
    #        a1.SetNumExplicitHs(int(explicitHs-1))
    #    RdkitMol_r1.UpdatePropertyCache(strict=True)
    mp2 = AllChem.MMFFGetMoleculeProperties(RdkitMol_r1)
    if mp2 == None: return
    if verbose: print("No. of atoms after fix = {:n}\n".format(RdkitMol_r1.GetNumAtoms()))
    # Print out all partial charges
    t=0.0
    for i in range(RdkitMol_r1.GetNumAtoms()):
        q = mp2.GetMMFFPartialCharge(i)
        t += q
        #print(q)
    if verbose: print("Total Charge after fix = {}\n".format(t))
    return RdkitMol_r1


def fix_charges(smiles, verbose=False):
    from io import StringIO
    import sys
    #smiles = "c1ccc(cc1)CC2=[NH2]N=C(S2)NC(=O)c3cccs3"
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    Chem.WrapLogs()
    sio = sys.stderr = StringIO()
    validate_rdkit_mol(mol)
    error = sio.getvalue()
    condition = True
    shift = 1
    while condition:
        if "Explicit valence for atom" in error:
            atomIdx = int(error.split("# ")[-1].split(" ")[0])
            atom = mol.GetAtomWithIdx(atomIdx)
            FC = atom.GetFormalCharge()
            if verbose: print(FC)
            if "greater than permitted" in error:
                new_mol = modify_charge(mol, atomIndex=atomIdx,
                        action=None, charge=FC, to=+shift, verbose=verbose)
            elif "less than permitted" in error:
                new_mol = modify_charge(mol, atomIndex=atomIdx,
                        action=None, charge=FC, to=-shift, verbose=verbose)
        else: print(error); return mol
        smiles = Chem.MolToSmiles(new_mol)
        if verbose: print(smiles)
        sio = sys.stderr = StringIO()
        if validate_rdkit_mol(new_mol): condition = False
        else: shift += 1
        error = sio.getvalue()
    return new_mol



def check_for_radicals(molecule):
    Bool = False
    if molecule != None:
        for atom in molecule.GetAtoms():
            if atom.GetNumRadicalElectrons() >= 1:
                Bool = True
                break
    return Bool





def get_mol_info(mol):
    """Returns information about the atoms including the index of each.
    """

    mol_info = []
    for i in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(i)
        info = dict()
        info["symbol"] = atom.GetSymbol()
        info["formal charge"] = atom.GetFormalCharge()
        info["neighbors"] = [a.GetSymbol() for a in atom.GetNeighbors()]
        info["neighbors Idx"] = [a.GetIdx() for a in atom.GetNeighbors()]
        info["nHydrogens"] = sum([1 for a in atom.GetNeighbors() if a.GetSymbol() == "H"])
        #info["nHydrogens"] = atom.GetTotalNumHs()
        mol_info.append(info)
    df = pd.concat([pd.DataFrame(mol_info), get_Morgan_fingerprints(mol)], axis=1)
    df = df.loc[:,~df.columns.duplicated()]
    return df


def check_transitions(microtransitions, df):
    """Returns microtransitions in correct order: deprot. to prot. """

    new = []
    for trans in microtransitions:
        if (list(df["nHydrogens"])[int(trans[0])] < list(df["nHydrogens"])[int(trans[1])]):
            new.append(trans)
        else:
            #print(f"{trans} --> {trans[::-1]}")
            new.append(trans[::-1])
    return np.array(new)


def mol_with_atom_index(molecule):
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    return molecule



#AllChem.MMFFOptimizeMoleculeConfs(m2, numThreads=0)


def get_num_ionizable_groups(db, smiles_key):
    """Given a pandas dataframe, Return the dataframe with a new column
    containing the number of ionizable groups for each molecule."""

    FPs = ["acidic", "basic", "acceptor", "donor"] # NOTE: fingerprints to look for :IMPORTANT:
    num_ionizable_groups = []
    for smiles in list(db[smiles_key]):
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        # get the molecular fingerprints from mol info
        mol_info = get_mol_info(mol)
        fp_atomIdxs = [row[0] for k,row in enumerate(mol_info.iterrows())
            if set(FPs).intersection(set(mol_info["Morgan fingerprints"][k]))]
        ionizable_sites = sum([1 for j in fp_atomIdxs if mol_info["nHydrogens"][j] > 0])
        #NOTE: uncomment for 0 --> 1
        #if ionizable_sites == 0: ionizable_sites = 1
        num_ionizable_groups.append(int(ionizable_sites))
    db["num ionizable groups"] = num_ionizable_groups
    return db

def get_mol_weight(db, smiles_key):
    weight = []
    for smiles in list(db[smiles_key]):
        weight.append(get_MolWt_from_smiles(smiles))
    db["Weight"] = weight
    return db


# ### Find all the repeats in the database
def find_duplicates(db, key1, key2):
    current1 = np.array([Chem.CanonSmiles(smiles) for smiles in np.array(list(db[key1]))])
    current2 = np.array([Chem.CanonSmiles(smiles) for smiles in np.array(list(db[key2]))])
    duplicates = []
    for i in range(len(list(db.index))):
        smiles1 = list(db[key1])[i]
        smiles2 = list(db[key2])[i]
        smiles1 = Chem.CanonSmiles(smiles1)
        smiles2 = Chem.CanonSmiles(smiles2)
        loc = np.where((smiles1 == current1) & (smiles2 == current2))[0]
        if len(loc) > 1:
            if duplicates != []:
                concat_dups = np.concatenate(duplicates)
                if loc[0] in concat_dups: continue
                else: duplicates.append(loc)
            else: duplicates.append(loc)
    return list(duplicates)


def curate_dataset(dataset, iterations=15):
    for i in range(iterations):
        dataset = drop_duplicates(dataset, key1='deprotonated microstate smiles',
                key2='protonated microstate smiles', threshold=0.03)
        dataset = dataset.reset_index(drop=True)
    # NOTE: Compute num of ionizable sites and molecular weight
    dataset = get_num_ionizable_groups(db=dataset, smiles_key="protonated microstate smiles")
    dataset = get_mol_weight(dataset, smiles_key='protonated microstate smiles')
    return dataset



def drop_duplicates(db, key1, key2, threshold=0.03, verbose=False):
    """ """

    dups = find_duplicates(db, key1, key2)
    print(f"Duplicates before dropping: {len(dups)}")
    for dup in dups:
        try: subset = db.iloc[dup]
        except(IndexError) as e: continue
        ID1 = list(subset[key1.replace("smiles","ID")])[0]
        ID2 = list(subset[key2.replace("smiles","ID")])[0]
        smiles1 = list(subset[key1])[0]
        smiles2 = list(subset[key2])[0]
        info = subset.describe()
        pKa_std = info["pKa"]["std"]
        if pKa_std > threshold:
            if len(subset) > 2:
                # find the row with the max deviation from the group of pKas
                # drop it from the list and calculate pKa std again
                while pKa_std > threshold:
                    if len(subset) == 2:
                        info = subset.describe()
                        pKa_std = info["pKa"]["std"]
                        if pKa_std > threshold:
                            index = list(subset.index)
                            db.drop(index=index, inplace=True)
                            break
                    pKas = list(subset["pKa"])
                    avg = np.average(list(pKas))
                    dev_dict = {list(subset["pKa"].index)[i]: abs(val - avg) for i,val in enumerate(pKas)}
                    index = max(dev_dict, key=dev_dict.get)
                    subset.drop(index=index, inplace=True) # drop all the values
                    db.drop(index=index, inplace=True)
                    info = subset.describe()
                    pKa_std = info["pKa"]["std"]
                    if pKa_std <= threshold:
                        avg_of_subset = subset.groupby([key1,key2]).mean()
                        avg_of_subset[key1.replace("smiles","ID")] = ID1
                        avg_of_subset[key2.replace("smiles","ID")] = ID2
                        avg_of_subset[key1] = smiles1
                        avg_of_subset[key2] = smiles2
                        db = pd.concat([db,avg_of_subset], axis=0)
                        index = list(subset.index)
                        db.drop(index=index, inplace=True)
            else:
                try: db.drop(index=dup, inplace=True) # drop all the values
                except(KeyError) as e: continue
        else:
            avg_of_subset = subset.groupby([key1,key2]).mean()
            avg_of_subset[key1.replace("smiles","ID")] = ID1
            avg_of_subset[key2.replace("smiles","ID")] = ID2
            avg_of_subset[key1] = smiles1
            avg_of_subset[key2] = smiles2
            db = pd.concat([db,avg_of_subset], axis=0)
            db.drop(index=dup, inplace=True) # drop all the values
    dups = find_duplicates(db, key1, key2)
    print(f"Duplicates after dropping: {len(dups)}")
    return db



def df_to_database(df, smiles_col="Smiles", verbose=False):
    """Remove a hydrogen and create the pKa database.
    """

    from .features import get_features
    #if isinstance(database, str):
    db = pd.DataFrame()
    FPs = ["acidic", "basic", "acceptor", "donor"] # NOTE: fingerprints to look for :IMPORTANT:
    for i,smiles in enumerate(df[smiles_col]):
        mol = Chem.MolFromSmiles(smiles)#, sanitize=True)
        #mol = Chem.AddHs(mol)
        macrostateID = list(df["Name"])[i]
        print(f"\n{i} Name: {macrostateID}\n")
        # NOTE: Criteria for finding the ionizable atom
        mol_info = get_mol_info(mol) # get the molecular fingerprints from mol info
        #print(mol_info)
        #exit()
        fp_atomIdxs = [row[0] for k,row in enumerate(mol_info.iterrows())
                if set(FPs).intersection(set(mol_info["Morgan fingerprints"][k]))]
        # NOTE: Ionizable sites:
        ionizable_sites = sum([1 for j in fp_atomIdxs if mol_info["nHydrogens"][j] > 0])
        #if ionizable_sites > 2: print(f"Number of ionizable sites: {ionizable_sites}\nContinue....");continue
        if smiles_col=="Ionizable center smiles":
            atomIdx = 0
            atomCharge = mol_info["formal charge"][atomIdx]
            temp_df = pd.DataFrame()
            protMol = mol
            Bool,temp_df = append_microstate_to_df(temp_df, smiles, macrostateID, microstateID=0)
            deprotMol = modify_charge(protMol, atomIndex=atomIdx, action=None, charge=atomCharge, to=int(atomCharge-1))
            # This code block is required to remove a Hydrogen
            a1 = deprotMol.GetAtomWithIdx(atomIdx)
            numHs = a1.GetTotalNumHs()
            if numHs != 0: a1.SetNumExplicitHs(int(numHs-1))
            deprotMol.UpdatePropertyCache(strict=True)
            Bool,temp_df = append_microstate_to_df(temp_df, Chem.MolToSmiles(deprotMol), macrostateID, microstateID=1)
            feat_df = get_features(temp_df, microtransitions=np.array([[0,1]]), verbose=verbose)
            feat_df["pKa"] = list(df["pKa"])[i]
            if "pKa source" in df.keys(): feat_df["pKa source"] = list(df["pKa source"])[i]
            feat_df["href"] = list(df["href"])[i]
            if "Weight" in df.keys(): feat_df["Weight"] = list(df["Weight"])[i]
            else: get_MolWt_from_smiles(smiles)
            mol_info = get_mol_info(protMol) # get the molecular fingerprints from mol info
            fp_atomIdxs = [row[0] for k,row in enumerate(mol_info.iterrows())
                if set(FPs).intersection(set(mol_info["Morgan fingerprints"][k]))]
            ionizable_sites = sum([1 for j in fp_atomIdxs if mol_info["nHydrogens"][j] > 0])
            if "Type" in df.keys(): feat_df["Type"] = list(df["Type"])[i]
            db = db.append(feat_df, ignore_index=True)
    db = get_num_ionizable_groups(db, smiles_key="protonated microstate smiles")
    return db



def append_microstate_to_df(df, smiles, macrostateID, microstateID):
    """Handy function that autmoatically checks the uniqueness of a microstate
    before appending it to the dataframe.
    """

    try:
        temp = df.to_dict('records')
    except(AttributeError) as e:
        print("Appending to empty DataFrame...")
        temp = []
    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    mol = Chem.AddHs(mol)
    mol_info = get_mol_info(mol)
    temp.append({"macrostate ID": macrostateID, "microstate ID": f"micro{microstateID:03d}", "smiles": smiles,
        "charge state": sum([mol.GetAtomWithIdx(i).GetFormalCharge() for i in range(mol.GetNumAtoms())]),
        "nHydrogens": sum([H for i,H in enumerate(mol_info["nHydrogens"])
            if (set(["basic","acceptor","donor"]).intersection(set(mol_info["Morgan fingerprints"][i])))
            or (mol_info["formal charge"][i] >= 1)]),
        })
    new_df = pd.DataFrame(temp)

    #NOTE: Check to make sure all of the smiles strings are following the same trends
    # so that the images will come out in the correct orientation
    endswith = new_df["smiles"][0][-3:]
    for i,smiles in enumerate(new_df["smiles"]):
        if smiles[-3:] != endswith:
            new_df["smiles"][i] = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))

    if toolbox.isnotebook(): PandasTools.AddMoleculeColumnToFrame(new_df, smilesCol='smiles', molCol='molecule')

    # check to make sure that the smile string is unique, otherwise don't append it
    try:
        count = list(new_df["smiles"].value_counts()[new_df["smiles"].value_counts() > 1])[0]
    except(IndexError):
        count = 1
    if count > 1:
        nonunique = list(new_df["smiles"].value_counts()[new_df["smiles"].value_counts() > 1].index)[0]
        match = np.where(np.array(nonunique)==np.array(list(new_df["smiles"])))
        print(f'\nWARNING: {nonunique} exists in pandas DataFrame at index {match[0][0]}!\n')
        return False,df
    else:
        if validate_rdkit_mol(mol):
            return True,new_df
        else:
            return False,df



def set_overlap_populations(mol):
    """Mulilken Overlap Population: Setting a property on each of the bonds
    """

    bond_info = []
    passed,res = rdEHTTools.RunMol(mol)
    energies = res.GetOrbitalEnergies()
    # Mulilken Overlap Population: Setting a property on each of the bonds
    rop = res.GetReducedOverlapPopulationMatrix()
    for bnd in mol.GetBonds():
        a1 = bnd.GetBeginAtom()
        a2 = bnd.GetEndAtom()
        i1 = max(a1.GetIdx(),a2.GetIdx())
        i2 = min(a1.GetIdx(),a2.GetIdx())
        idx = (i1*(i1+1))//2 + i2
        bnd.SetDoubleProp("MullikenOverlapPopulation",rop[idx])
        pop = bnd.GetDoubleProp("MullikenOverlapPopulation")
        bond_info.append({
            "begin atom symbol": bnd.GetBeginAtom().GetSymbol(),
            "begin atom Idx": bnd.GetBeginAtomIdx(),
            "end atom symbol": bnd.GetEndAtom().GetSymbol(),
            "end atom Idx": bnd.GetEndAtomIdx(),
            "Mulliken population": pop,
            })
    df = pd.DataFrame(bond_info)
    return df


###############################################################################
#NOTE:
# get minimization
# https://nbviewer.jupyter.org/github/iwatobipen/playground/blob/master/openff.ipynb
# get fractional bond orders
# http://rdkit.blogspot.com/2019/06/doing-extended-hueckel-calculations.html
###############################################################################
#from openforcefield.topology import Molecule
#https://open-forcefield-toolkit.readthedocs.io/en/topology/api/generated/openforcefield.topology.Molecule.html#openforcefield.topology.Molecule.compute_wiberg_bond_orders
#molecule.compute_wiberg_bond_orders() # NOTE: Doesn't work...
# Wiberg bond order is a measure of electron population overlap between two atoms:
# W_{AB} = \sum\limits_{\mu \in  A} \sum\limits_{\nu \in B} P_{\mu\nu}^2
#


#def get_minimized_structures(df, verbose=False):
#
#    for i in range(len(list(df["microstate ID"]))):
#        smiles = df["smiles"][i]
#        mol = Chem.MolFromSmiles(smiles, sanitize=True)
#        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
#        mol_info = get_mol_info(mol)
#        if verbose: print(f'Computing partial charges for\
#                {df["macrostate ID"][i]}_{df["microstate ID"][i]}\n{mol_info}')
#        mh_mmff = minimize_structure(mol, maxIters=5000, nConfs=100)


def get_conformer_energies(mol, mmffVariant="MMFF94s"):
    energies = []
    for conf in mol.GetConformers():
        tm = Chem.Mol(mol, False, conf.GetId())
        prop = AllChem.MMFFGetMoleculeProperties(tm, mmffVariant) # MMFF94
        ff = AllChem.MMFFGetMoleculeForceField(tm, prop)
        energy = ff.CalcEnergy() # kcal/mol
        energies.append(energy)
    energies = np.asarray(energies, dtype=float)
    return energies


def minimize_structure(molecule, maxIters=500, nConfs=50, mmffVariant="MMFF94s",
        filename=None, verbose=False):
    """
    https://www.rdkit.org/docs/source/rdkit.Chem.rdForceFieldHelpers.html#rdkit.Chem.rdForceFieldHelpers.MMFFOptimizeMoleculeConfs
    """


    remover = SaltRemover.SaltRemover(defnData="[Na,Cl,Br]")
    molecule = remover.StripMol(molecule)

    m = Chem.AddHs(molecule)
    try:
        AllChem.EmbedMultipleConfs(m, nConfs);
    except(ValueError,AttributeError) as e:
        print(e)
        m = fix_charges(Chem.MolToSmiles(m), verbose=False)
        AllChem.EmbedMultipleConfs(m, nConfs=nConfs, numThreads=0);
    passed,res = rdEHTTools.RunMol(m)
    if passed:
        m_mmff = Chem.Mol(m)
        AllChem.MMFFOptimizeMoleculeConfs(m_mmff, maxIters=maxIters,
                numThreads=0, mmffVariant=mmffVariant)
    else:
        print("ERROR: Unable to minimize...")
        raise RuntimeError

    try:
        energies = get_conformer_energies(mol=m_mmff, mmffVariant=mmffVariant)
        sorted_energies = np.argsort(energies)
        conf_ids = [conf.GetId() for conf in m_mmff.GetConformers()]
        result = Chem.Mol(m_mmff)
        result.RemoveAllConformers()
        for i in sorted_energies:
            conf = m.GetConformer(conf_ids[i])
            result.AddConformer(conf, assignId=True)
        if filename != None:
            Chem.rdmolfiles.MolToPDBFile(m_mmff, filename=filename.replace(".mol","_ensemble.pdb"))
            Chem.rdmolfiles.MolToMolFile(result, confId=0, filename=filename)
            Chem.rdmolfiles.MolToPDBFile(result, confId=0, filename=filename.replace(".mol",".pdb"))
    except (Exception) as e:
        print(e)
        print("Can't get energies. Returning unsorted conformations...")
        result = Chem.Mol(m_mmff)
    return result


def compute_EH(molecule, atoms="heavy"):
    """Calculate Extended Hückel Molecular orbital method of a molecule's conformers
    and return a Pandas DataFrame of the results.
    Returns the average partial charge over all the conformers
    """

    df,eres = [],[]
    if atoms == "heavy":
        charges = [[] for x in range(molecule.GetNumHeavyAtoms())]
    if atoms == "all":
        charges = [[] for x in range(molecule.GetNumAtoms())]
    for ID in range(molecule.GetNumConformers()):
        passed,res = rdEHTTools.RunMol(molecule, confId=ID)
        eres.append(res)
        if not passed:
            raise ValueError("Extended Hückel tools failed to load in molecule...")
        hvyIdx = 0
        echgs = res.GetAtomicCharges()
        for atom in molecule.GetAtoms():
            if atoms == "heavy":
                if atom.GetAtomicNum()==1:
                    continue
            charges[hvyIdx].append(echgs[atom.GetIdx()])
            hvyIdx+=1
    avg_charge = np.average(charges, axis=1)
    std_charge = np.std(charges, axis=1)
    hvyIdx = 0
    for atom in molecule.GetAtoms():
        if atoms == "heavy":
            if atom.GetAtomicNum()==1:
                continue
        atom.SetDoubleProp("ExtendedHuckelPartialCharge", avg_charge[hvyIdx])
        pc = atom.GetDoubleProp("ExtendedHuckelPartialCharge")

        df.append({"Symbol": atom.GetSymbol(), #"atomIdx": atom.GetIdx(),
            "partial charge": avg_charge[hvyIdx],
            "std partial charge": std_charge[hvyIdx]})

        hvyIdx+=1
    df = pd.DataFrame(df)
    #return (eres,df)
    return df


def compute_AM1BCC_charges(molecule, atoms="heavy"):
    """Calculate AM1BCC charges"""

    from openforcefield.topology import Molecule
    #https://open-forcefield-toolkit.readthedocs.io/en/topology/api/generated/openforcefield.topology.Molecule.html#openforcefield.topology.Molecule.compute_wiberg_bond_orders
    molecule = Molecule.from_rdkit(molecule, allow_undefined_stereo=True)
    molecule.compute_partial_charges_am1bcc()
    #molecule.compute_wilberg_bond_orders(charge_model="AM1")
    #help(molecule)
    m = molecule.to_rdkit()
    molecule = molecule.to_rdkit()
    df,props = [],[]
    for i in range(m.GetNumAtoms()):
        if atoms=="heavy":
            if m.GetAtomWithIdx(i).GetAtomicNum()!=1:
                props.append(m.GetAtomWithIdx(i).GetPropsAsDict())
        elif atoms=="all":
            props.append(m.GetAtomWithIdx(i).GetPropsAsDict())
        else:
            print('ERROR: atoms="heavy" or atoms="all"')
            exit()
    props = pd.DataFrame(props)
    hvyIdx = 0
    for atom in m.GetAtoms():
        if atoms == "heavy":
            if atom.GetAtomicNum()==1:
                continue
        #atom.SetDoubleProp("AM1BCCPartialCharge", props["partial_charge"][hvyIdx])
        #pc = atom.GetDoubleProp("AM1BCCPartialCharge")
        df.append({"Symbol": atom.GetSymbol(), #"atomIdx": atom.GetIdx(),
            "partial charge": props["partial_charge"][hvyIdx],})
            #"std partial charge": props["partial_charge"][hvyIdx]
        hvyIdx+=1
    return pd.DataFrame(df)


def compute_Gasteiger_charges(molecule, atoms="heavy"):
    """Calculate AM1BCC charges"""

    Chem.rdPartialCharges.ComputeGasteigerCharges(molecule)
    df,props = [],[]
    for i in range(molecule.GetNumAtoms()):
        if atoms=="heavy":
            if molecule.GetAtomWithIdx(i).GetAtomicNum()!=1:
                props.append(molecule.GetAtomWithIdx(i).GetPropsAsDict())
        elif atoms=="all":
            props.append(molecule.GetAtomWithIdx(i).GetPropsAsDict())
        else:
            print('ERROR: atoms="heavy" or atoms="all"')
            exit()
    props = pd.DataFrame(props)
    hvyIdx = 0
    for atom in molecule.GetAtoms():
        if atoms == "heavy":
            if atom.GetAtomicNum()==1:
                continue
        df.append({"Symbol": atom.GetSymbol(), #"atomIdx": atom.GetIdx(),
            "partial charge": props["_GasteigerCharge"][hvyIdx],})
            #"std partial charge": props["partial_charge"][hvyIdx]
        hvyIdx+=1
    return pd.DataFrame(df)



#def calculate_partial_charges(molecule, atoms="heavy", maxIters=500, nConfs=50, method="AM1BCC"):
#    """Uses three methods to compute partial charges:
#        AM1BCC, Gasteiger, Extented Hückel
#    Args:
#        method(str) - "AM1BCC", "Gasteiger", "Extented Hückel", "all"
#
#    """
#
#    mh_mmff = minimize_structure(molecule, maxIters, nConfs)
#
#    if method=="Extented Hückel": return compute_EH(mh_mmff, atoms)
#    if method=="AM1BCC": return compute_AM1BCC_charges(mh_mmff, atoms)
#    if method=="Gasteiger":
#        Gast = compute_Gasteiger_charges(molecule, atoms)
#        Gast.columns = ["Symbol", "Gasteiger"]
#        return Gast
#
#    if method=="all":
#        EH = compute_EH(mh_mmff, atoms)
#        EH.columns = ["Symbol", "Extended Hückel", "Extended Hückel std"]
#        AM1 = compute_AM1BCC_charges(mh_mmff, atoms)
#        AM1.columns = ["Symbol", "AM1BCC"]
#        if atoms=="heavy":
#            Gast = compute_Gasteiger_charges(molecule, atoms)
#            Gast.columns = ["Symbol", "Gasteiger"]
#            df = pd.concat([EH, AM1, Gast], axis=1)
#        else:
#            df = pd.concat([EH, AM1], axis=1)
#        df = df.loc[:,~df.columns.duplicated()]
#        return df

def get_bond_order(molecule):
    """Calculate AM1BCC charges"""

    from openforcefield.topology import Molecule
    #https://open-forcefield-toolkit.readthedocs.io/en/topology/api/generated/openforcefield.topology.Molecule.html#openforcefield.topology.Molecule.compute_wiberg_bond_orders
    molecule = Molecule.from_rdkit(molecule, allow_undefined_stereo=True)
    molecule.generate_conformers()
    fbo = molecule.get_fractional_bond_orders()
    return fbo # FIXME



def get_partial_charges(df, method="AM1BCC", atoms="heavy", verbose=False):
    """
    Args:
        method(str) - "AM1BCC", "Gasteiger", "Extented Hückel"
    """

    result = pd.DataFrame()
    print(df)
    exit()
    for i in range(len(list(df["microstate ID"]))):
        smiles = df["smiles"][i]
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
        mol_info = get_mol_info(mol)
        if verbose:
            print(f'Computing partial charges for {df["macrostate ID"][i]}_{df["microstate ID"][i]}')
            print(mol_info)

        # Get all the partial charges
        try:
            mh_mmff = minimize_structure(mol, maxIters=5000, nConfs=100)
            if method=="Extented Hückel": pc = compute_EH(mh_mmff, atoms)
            if method=="AM1BCC": pc = compute_AM1BCC_charges(mh_mmff, atoms)
            if method=="Gasteiger":
                pc = compute_Gasteiger_charges(molecule, atoms)
                pc.columns = ["Symbol", "Gasteiger"]
        except(Exception,ValueError) as e:
            # if openforcefield has issues, then we need a work-around. Point to OpenEye
            print(e)
            pc = OE.calculate_partial_charges(smiles)

        name = f'{df["macrostate ID"][i]}_{df["microstate ID"][i]}'
        pc.columns = ["Symbol", name]
        result = pd.concat([result, pc], axis=1)
    result = result.loc[:,~result.columns.duplicated()]
    return result


#def get_partial_charges(df, method="AM1BCC", atoms="heavy", verbose=False):
#    """
#    Args:
#        method(str) - "AM1BCC", "Gasteiger", "Extented Hückel"
#    """
#
#    result = pd.DataFrame()
#    for i in range(len(list(df["microstate ID"]))):
#        smiles = df["smiles"][i]
#        mol = Chem.MolFromSmiles(smiles, sanitize=True)
#        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
#        mol_info = get_mol_info(mol)
#        if verbose:
#            print(f'Computing partial charges for {df["macrostate ID"][i]}_{df["microstate ID"][i]}')
#            print(mol_info)
#
#        # Get all the partial charges
#        try:
#            mh_mmff = minimize_structure(molecule, maxIters=5000, nConfs=100)
#            if method=="Extented Hückel": pc = compute_EH(mh_mmff, atoms)
#            if method=="AM1BCC": pc = compute_AM1BCC_charges(mh_mmff, atoms)
#            if method=="Gasteiger":
#                pc = compute_Gasteiger_charges(molecule, atoms)
#                pc.columns = ["Symbol", "Gasteiger"]
#        except(Exception,ValueError) as e:
#            # if openforcefield has issues, then we need a work-around. Point to OpenEye
#            print(e)
#            pc = OE.calculate_partial_charges(smiles)
#
#        name = f'{df["macrostate ID"][i]}_{df["microstate ID"][i]}'
#        pc.columns = ["Symbol", name]
#        result = pd.concat([result, pc], axis=1)
#    result = result.loc[:,~result.columns.duplicated()]
#    return result




#def get_partial_charges(df, method="AM1BCC", verbose=False):
#    """
#    Args:
#        method(str) - "AM1BCC", "Gasteiger", "Extented Hückel"
#    """
#
#    result = pd.DataFrame()
#    for i in range(len(list(df["microstate ID"]))):
#        smiles = df["smiles"][i]
#        mol = Chem.MolFromSmiles(smiles, sanitize=True)
#        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
#        mol_info = get_mol_info(mol)
#        if verbose:
#            print(f'Computing partial charges for {df["macrostate ID"][i]}_{df["microstate ID"][i]}')
#            print(mol_info)
#
#        # Get all the partial charges
#        try:
#            pc = calculate_partial_charges(mol, atoms="heavy", maxIters=5000, nConfs=100, method=method)
#        except(Exception,ValueError) as e:
#            # if openforcefield has issues, then we need a work-around. Point to OpenEye
#            print(e)
#            pc = OE.calculate_partial_charges(smiles)
#
#        #if outpath != None:
#        #    ax = pc[method].plot(kind="line", label=method)
#        #    ax.legend()
#        #    fig = ax.get_figure()
#        #    figname = os.path.join(f'{outpath}',f'partial_charges_{df["macrostate ID"][i]}_{df["microstate ID"][i]}.png')
#        #    fig.savefig(figname)
#        #    plt.close(fig)
#
#        name = f'{df["macrostate ID"][i]}_{df["microstate ID"][i]}'
#        pc.columns = ["Symbol", name]
#        result = pd.concat([result, pc], axis=1)
#    result = result.loc[:,~result.columns.duplicated()]
#    return result




def get_micro_transitions(df):
    """Note that this is an exaustive list of all microstate transitions and
    will be more than needed...
    """
    #TODO: criteria for 2^n  microtransitions (thermodynamic cycle)

   # Direction is from protonated to deprotonated
    df = df.sort_values('nHydrogens', ascending=False)
    combos = list(combinations(df.index,2))
    transitions = []
    for combo in combos:
        if abs(df["nHydrogens"][combo[0]] - df["nHydrogens"][combo[1]]) == 1:
            transitions.append(combo)
        else:
            continue
    return np.array(transitions)
    #from_list, to_list = np.array(final).transpose()
    #from_list = [str(df["macrostate ID"][i])+"_"+str(df["microstate ID"][i]) for i in from_list]
    #to_list = [str(df["macrostate ID"][i])+"_"+str(df["microstate ID"][i]) for i in to_list]
    #micros = sorted(set(np.concatenate([to_list, from_list])), key=lambda x:[convert(s) for s in re.split("([0-9]+)",x)])

def calc_SASA(mol, atom_indices, alg="Shrake"):
    """alg(str) - Shrake or Lee
    """

    radii = rdFreeSASA.classifyAtoms(mol)
    if alg == "Lee": alg = rdFreeSASA.SASAAlgorithm.LeeRichards
    if alg == "Shrake": alg = rdFreeSASA.SASAAlgorithm.ShrakeRupley
    opts = rdFreeSASA.SASAOpts()
    opts.algorithm = alg
    sasa = rdFreeSASA.CalcSASA(mol, radii, confIdx=0, opts=opts)
    result = [float(a.GetProp("SASA")) for a in mol.GetAtoms() if a.GetIdx() in atom_indices]
    #print([(a.GetSymbol(),a.GetIdx()) for a in mol.GetAtoms() if a.GetIdx() in atom_indices])
    return result


def get_SASA(filename, atom_indices=None):

    filename = os.path.abspath(filename)
    mol2 = md.load(filename)
    sasa = md.shrake_rupley(mol2)
    if atom_indices == None:
        return sasa
    else:
        return [sasa[0][Idx] for Idx in atom_indices]


#if __name__ == "__main__":
#    smiles = "c1(C[NH2+]c(nc[nH+]2)c3c2cccc3)ccccc1"
#    get_SASA(smiles, atom_indices=[5,7,9])
#



#def AddH(mol, atomIdx):
#
#    a = mol.GetAtomWithIdx(atomIdx)
#    mol = oechem.OEAddExplicitHydrogens(mol, a)
#    return mol
#
#
#def delete_atom(mol, atomIdx):
#
#    atom = list(mol.GetAtoms())[atomIdx]
#    mol.DeleteAtom(atom)
#    return mol
#
#def protonate_atom(mol, atomIdx):
#
#    atom = list(mol.GetAtoms())[atomIdx]
#    H = mol.NewAtom(1)
#    mol.NewBond(atom, H)
#    return mol





