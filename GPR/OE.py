import sys, os
import pandas as pd
from . import toolbox
from . import RD
from openeye import oechem
from openeye import oeomega
from openeye import oequacpac
from openeye import oeszybki
from openeye import oespicoli
from openeye import oeszmap
from openeye import oeff
import ctypes
import numpy as np
import rdkit
from rdkit import Chem

if not oechem.OEChemIsLicensed("python"):
    oechem.OEThrow.Warning("OEChem is not licensed for the python feature")
expdate = oechem.OEUIntArray(3)
if oechem.OEChemIsLicensed("python", expdate):
    oechem.OEThrow.Info("License expires: day: %d  month: %d year: %d"
                        % (expdate[0], expdate[1], expdate[2]))


def get_SASA(mol):
    """
    https://docs.eyesopen.com/toolkits/python/spicolitk/OESpicoliFunctions/OEMakeAccessibleSurface.html#OESpicoli::OEMakeAccessibleSurface
    """
    pass # NOTE: Doesn't work yet. (I'm using MDTraj to do this...
    surf = oespicoli.OESurface()
    help(oespicoli.OEMakeAccessibleSurface)
    oespicoli.OEMakeAccessibleSurface(surf, coords, radii, natoms=1)
    SASA = oespicoli.OESurfaceArea(surf)
    return SASA


def get_thermodynamics(filename, verbose=False):
    """
    https://docs.eyesopen.com/toolkits/python/
    """

    ifs = oechem.oemolistream()
    if not ifs.open(filename):
        oechem.OEThrow.Fatal("Unable to open %s for reading" % filename)

    mol = oechem.OEMol()
    oechem.OEReadMolecule(ifs, mol)

    opts = oeszybki.OESzybkiOptions()
    opts.GetGeneralOptions().SetForceFieldType(oeszybki.OEForceFieldType_MMFF94)
    sz = oeszybki.OESzybki(opts)
    results = oeszybki.OESzybkiResults()
    if not sz(mol, results):
        return 1

    eresults = oeszybki.OESzybkiEnsembleResults()
    entropy = sz.GetEntropy(mol, eresults, oeszybki.OEEntropyMethod_Analytic, oeszybki.OEEnvType_Gas)
    if verbose:
        print(f"Estimated molar solution entropy of the input compound: {entropy} J/(mol K)")
        results.Print(oechem.oeout)
    temperature = eresults.GetTemperature()
    total_energy = results.GetTotalEnergy()
    if verbose:
        print(f"Temperature: {temperature} K")
        print(f"Total energy: {total_energy} kcal")
        print(f"Entropic Energy: {eresults.GetEntropicEnergy()} kcal/(mol)")
        print(f"Total entropy: {eresults.GetTotalEntropy()} J/(mol K)")
        print(f"Estimated molar solution entropy of the input compound: {entropy} J/(mol K)")
    G = [r.GetConfFreeEnergyFromEnsemble() for r in eresults.GetResultsForConformations()]
    free_energy = np.mean(G) # Should be 0 since it is in vacuum...
    return entropy,free_energy



def get_Mayer_partial_bond_order(filename, verbose=False):
    """
    https://docs.eyesopen.com/toolkits/python/
    """

    bond_info = []
    ifs = oechem.oemolistream()
    if not ifs.open(filename):
        oechem.OEThrow.Fatal("Unable to open %s for reading" % filename)
    am1 = oequacpac.OEAM1()
    results = oequacpac.OEAM1Results()
    for mol in ifs.GetOEMols():
        for conf in mol.GetConfs():
            print("molecule: ", mol.GetTitle(), "conformer:", conf.GetIdx())
            if am1.CalcAM1(results, mol):
                nbonds = 0
                for bond in mol.GetBonds(oechem.OEIsRotor()):
                    nbonds += 1
                    print(results.GetBondOrder(bond.GetBgnIdx(), bond.GetEndIdx()))
                    bond_info.append({
                        "begin atom symbol": bond.GetBgn(),#.GetName(),
                        "begin atom Idx":    bond.GetBgnIdx(),
                        "end atom symbol":   bond.GetEnd(),#.GetName(),
                        "end atom Idx":      bond.GetEndIdx(),
                        "Mayer Bond Order":  results.GetBondOrder(bond.GetBgnIdx(), bond.GetEndIdx()),
                        })
    df = pd.DataFrame(bond_info)
    return df


def optimize_SM(smiles, filename="temp.mol2", maxConfs=1, verbose=False):
    """
    The following function is an example to show how to evaluate energy
    and optimize a small molecule, with keeping a subset of atoms fixed,
    using the MMFF force field.
    https://docs.eyesopen.com/toolkits/python/
    """




    smiles = set_stereochemistry(smiles)
    write_mol_file_from_smiles(smiles, filename, maxConfs)

    ifs = oechem.oemolistream()
    if not ifs.open(filename):
        oechem.OEThrow.Fatal("Unable to open %s for reading" % filename)

    ofs = oechem.oemolostream()
    outname = str(filename.split(".")[0]+"_op.mol2")
    if not ofs.open(outname):
        oechem.OEThrow.Fatal("Unable to open %s for writing" % outname)

    mol = oechem.OEMol()
    while oechem.OEReadMolecule(ifs, mol):
        # Set stereochemistry
        for atom in mol.GetAtoms():
            if atom.IsChiral() and not atom.HasStereoSpecified(oechem.OEAtomStereo_Tetrahedral):
                v = []
                for neigh in atom.GetAtoms():
                    v.append(neigh)
                atom.SetStereo(v, oechem.OEAtomStereo_Tetra, oechem.OEAtomStereo_Left)

        oechem.OESetDimensionFromCoords(mol)
        mol.SetDimension(3)
        oechem.OEAddExplicitHydrogens(mol)

        mmff = oeff.OEMMFF()
        mmff = oeff.OEGenericFF(mmff)
        #mmff = oeff.OEMMFFAmber()
        if not mmff.PrepMol(mol) or not mmff.Setup(mol):
            oechem.OEThrow.Warning("Unable to process molecule: title = '%s'" % mol.GetTitle())
            oechem.OEWriteMolecule(ofs, mol)
            continue

        #oechem.OEMolBase.SetDimension(3)
        vecCoords = oechem.OEDoubleArray(3*mol.GetMaxAtomIdx())
        for conf in mol.GetConfs():
            if verbose:
                oechem.OEThrow.Info("Molecule: %s Conformer: %d" % (mol.GetTitle(), conf.GetIdx()+1))
            conf.GetCoords(vecCoords)

            energy = mmff(vecCoords)
            if verbose:
                oechem.OEThrow.Info("Initial energy: %0.2f kcal/mol" % energy)

            optimizer = oeff.OEBFGSOpt()
            energy = optimizer(mmff, vecCoords, vecCoords)
            if verbose:
                oechem.OEThrow.Info("Optimized energy: %0.2f kcal/mol" % energy)
            conf.SetCoords(vecCoords)

        oechem.OEWriteMolecule(ofs, mol)
    return outname




def get_entropy(smiles, filename="temp.mol2", verbose=False):
    """
    https://docs.eyesopen.com/toolkits/python/
    """

    #write_mol_file_from_smiles(smiles, filename)
    filename = optimize_SM(smiles, filename="temp.mol2", verbose=verbose)

    ifs = oechem.oemolistream()
    if not ifs.open(filename):
        oechem.OEThrow.Fatal("Unable to open %s for reading" % filename)

    mol = oechem.OEMol()
    oechem.OEReadMolecule(ifs, mol)

    opts = oeszybki.OESzybkiOptions()
    opts.GetGeneralOptions().SetForceFieldType(oeszybki.OEForceFieldType_MMFF94)
    sz = oeszybki.OESzybki(opts)
    results = oeszybki.OESzybkiResults()
    if not sz(mol, results):
        return 1

    #oechem.OEWriteMolecule(ofs, mol)
    #free_energy = results.GetConfFreeEnergyFromEnsemble()
    eresults = oeszybki.OESzybkiEnsembleResults()
    entropy = sz.GetEntropy(mol, eresults, oeszybki.OEEntropyMethod_Analytic, oeszybki.OEEnvType_Gas)/1000 # convert to kJ/(mol K)
    if verbose:
        print(f"Estimated molar solution entropy of the input compound: {entropy} kJ/(mol K)")
        results.Print(oechem.oeout)
    temperature = eresults.GetTemperature()
    total_energy = results.GetTotalEnergy()
    #entropy = eresults.GetEntropicEnergy() # kcal/mol
    return entropy



def get_freeE_solv(filename, total_charge, verbose=False):
    """Returns the free energy of solvation from OpenEye.  The calculation uses
    a continuum solvent model (Poisson-Boltzmann)
    AM1BCC partial charges is used for the
    performed on conformation with the lowest energy
    :math:`\Delta G_{solv} = \sum_{i}^{n}(V_{i,solv} - V_{i,vac})q_{i}+S_{i}`
    :ref:`https://docs.eyesopen.com/toolkits/python/szybkitk/freeformtheory.html#solvation-energy-estimation`
    """

    #write_mol_file_from_smiles(smiles, filename)
    #filename = optimize_SM(smiles, filename="temp.mol2", maxConfs=200, verbose=verbose)

    ifs = oechem.oemolistream()
    if not ifs.open(filename):
        oechem.OEThrow.Fatal("Unable to open %s for reading" % filename)

    mol = oechem.OEMol()
    oechem.OEReadMolecule(ifs, mol)

    #mol = optimize(mol, maxConfs=500, verbose=verbose)

    # generate ensemble of conformations
    opts = oeszybki.OEFreeFormSolvOptions()
    opts.SetUseInputEnsemble(True)
    opts.SetUseInput3D(True)
    if total_charge == 0:
        opts.SetIonicState(oeszybki.OEFreeFormIonicState_Uncharged)
    else:
        opts.SetIonicState(oeszybki.OEFreeFormIonicState_PH74)
        #opts.SetIonicState(ctypes.c_uint(total_charge).value)
    print(f"Ionic state: {opts.GetIonicState()}")

    res = oeszybki.OEFreeFormSolvResults()

    omol = oechem.OEGraphMol()
    if not oeszybki.OEEstimateSolvFreeEnergy(res, omol, mol, opts):
        oechem.OEThrow.Error("Failed to calculate solvation free energy for molecule %s" %
                             mol.GetTitle())
    solvenergy = res.GetSolvationFreeEnergy()*4.184 # converting to kJ/mol
    if verbose:
        oechem.OEThrow.Info("Solvation free energy for compound %s is %6.4f kJ/mol" %
                            (mol.GetTitle(), solvenergy))
    #if filename == "temp.mol":
    #    os.remove(filename)
    return solvenergy


def set_stereochemistry(smiles):
    """Sets the sterochemistry for a molecule given a smiles string. Returns a
    smiles string.
    """

    mol = oechem.OEGraphMol()
    oechem.OESmilesToMol(mol, smiles)
    for atom in mol.GetAtoms():
        if atom.IsChiral() and not atom.HasStereoSpecified(oechem.OEAtomStereo_Tetrahedral):
            v = []
            for neigh in atom.GetAtoms():
                v.append(neigh)
            atom.SetStereo(v, oechem.OEAtomStereo_Tetra, oechem.OEAtomStereo_Left)
    return oechem.OEMolToSmiles(mol)


def write_mol_file_from_smiles(smiles, filename, maxConfs=200):

    mol_from_smiles = oechem.OEMol()
    oechem.OEParseSmiles(mol_from_smiles, smiles)
    mol = oechem.OEMol(mol_from_smiles)
    omega = oeomega.OEOmega()
    omega.SetMaxConfs(maxConfs)
    omega.SetStrictStereo(False) #NOTE:
    omega.SetStrictAtomTypes(False)
    omega(mol)
    ostream = oechem.oemolostream(filename)
    #Now we write our molecule
    oechem.OEWriteMolecule(ostream, mol)
    ostream.close()
    return 0



def AssignChargesByName(mol, name):
    if name == "noop":
        return oequacpac.OEAssignCharges(mol, oequacpac.OEChargeEngineNoOp())
    elif name == "mmff" or name == "mmff94":
        return oequacpac.OEAssignCharges(mol, oequacpac.OEMMFF94Charges())
    elif name == "am1bcc":
        return oequacpac.OEAssignCharges(mol, oequacpac.OEAM1BCCCharges())
    elif name == "am1bccnosymspt":
        optimize = True
        symmetrize = True
        return oequacpac.OEAssignCharges(mol, oequacpac.OEAM1BCCCharges(not optimize, not symmetrize))
    elif name == "amber" or name == "amberff94":
        return oequacpac.OEAssignCharges(mol, oequacpac.OEAmberFF94Charges())
    elif name == "am1bccelf10":
        return oequacpac.OEAssignCharges(mol, oequacpac.OEAM1BCCELF10Charges())
    return False


def calculate_partial_charges(smiles, atoms="heavy", maxIters=500, nConfs=50):
    """Uses three methods to compute partial charges:
        AM1BCC, Gasteiger, Extented Hückel
    Args:
        method(str) - "AM1BCC", "Gasteiger", "Extented Hückel", "all"
    """

    assign_charges(smiles, method="am1bcc")
    m = rdkit.Chem.rdmolfiles.MolFromMol2File("temp_op.mol2")
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
    #print(props)
    hvyIdx = 0
    for atom in m.GetAtoms():
        if atoms == "heavy":
            if atom.GetAtomicNum()==1:
                continue
        df.append({"Symbol": atom.GetSymbol(), #"atomIdx": atom.GetIdx(),
            "partial charge": props["_TriposPartialCharge"][hvyIdx],})
            #"std partial charge": props["partial_charge"][hvyIdx]
        hvyIdx+=1
    return pd.DataFrame(df)



def assign_charges(smiles, method="am1bcc", verbose=False):

    method = "am1bcc"
    ifs = oechem.oemolistream()
    if not ifs.open(filename):
        oechem.OEThrow.Fatal("Unable to open %s for reading" % filename)
    mol = oechem.OEMol()
    ofs = oechem.oemolostream()
    while oechem.OEReadMolecule(ifs, mol):
        if not AssignChargesByName(mol, chargeName):
            oechem.OEThrow.Warning("Unable to assign %s charges to mol %s"
                                   % (chargeName, mol.GetTitle()))
        oechem.OEWriteMolecule(ofs, mol)
    ifs.close()
    ofs.close()



def get_reasonable_charge_state(input_file, output_file):

    ifs = oechem.oemolistream()
    if not ifs.open(input_file):
        oechem.OEThrow.Fatal("Unable to open %s for reading" % input_file)

    ofs = oechem.oemolostream()
    if not ofs.open(output_file):
        oechem.OEThrow.Fatal("Unable to open %s for writing" % output_file)

    for mol in ifs.GetOEGraphMols():
        oequacpac.OEGetReasonableProtomers(mol)
        oechem.OEWriteMolecule(ofs, mol)



def enumerate_tautomers(input_file, output_file):
    """
    https://docs.eyesopen.com/toolkits/python/quacpactk/\
    examples_summary_enumeratetautomers.html#section-examples-summary-enumeratetautomers
    """

    ifs = oechem.oemolistream()
    if not ifs.open(input_file):
        oechem.OEThrow.Fatal("Unable to open %s for reading" % input_file)

    ofs = oechem.oemolostream()
    if not ofs.open(output_file):
        oechem.OEThrow.Fatal("Unable to open %s for writing" % output_file)

    tautomerOptions = oequacpac.OETautomerOptions()
    pKaNorm = True

    for mol in ifs.GetOEGraphMols():
        for tautomer in oequacpac.OEGetReasonableTautomers(mol, tautomerOptions, pKaNorm):
            oechem.OEWriteMolecule(ofs, tautomer)



###############################################################################
# Documentation:
###############################################################################
# NOTE: structure from smile string
# https://docs.eyesopen.com/toolkits/python/oechemtk/molctordtor.html#construction-from-smiles
# NOTE: identifying groups
# https://docs.eyesopen.com/toolkits/python/oechemtk/predicates.html#user-defined-functors
###############################################################################


def get_Wiberg_bond_order(smiles, filename="temp.mol2"):
    write_mol_file_from_smiles(smiles, filename)

    ifs = oechem.oemolistream()
    if not ifs.open(filename):
        oechem.OEThrow.Fatal("Unable to open %s for reading" % filename)

    mol = oechem.OEMol()
    oechem.OEReadMolecule(ifs, mol)

    if len(argv) != 2:
        oechem.OEThrow.Usage("%s <infile>" % argv[0])

    ifs = oechem.oemolistream(sys.argv[1])
    am1 = oequacpac.OEAM1()
    results = oequacpac.OEAM1Results()
    for mol in ifs.GetOEMols():
        for conf in mol.GetConfs():
            print("molecule: ", mol.GetTitle(), "conformer:", conf.GetIdx())
            if am1.CalcAM1(results, mol):
                nbonds = 0
                for bond in mol.GetBonds(oechem.OEIsRotor()):
                    nbonds += 1
                    print(results.GetBondOrder(bond.GetBgnIdx(), bond.GetEndIdx()))
                print("Rotatable bonds: ", nbonds)




def database_stuff():
    itf = oechem.OEInterface(InterfaceData, argv)

    # input - preserve rotor-offset-compression
    ifs = oechem.oemolistream()
    oechem.OEPreserveRotCompress(ifs)
    if not ifs.open(itf.GetString("-in")):
        oechem.OEThrow.Fatal("Unable to open %s for reading" % itf.GetString("-in"))

    # output - use PRE-compress for smaller files (no need to .gz the file)
    ofs = oechem.oemolostream()
    oechem.OEPRECompress(ofs)
    if not ofs.open(itf.GetString("-out")):
        oechem.OEThrow.Fatal("Unable to open '%s' for writing" % itf.GetString("-out"))
    if itf.GetString("-out").endswith('.gz'):
        oechem.OEThrow.Fatal("Output file must not gzipped")

    maxConfs = itf.GetInt("-maxConfs")
    if maxConfs < 1:
        oechem.OEThrow.Fatal("Illegal number of conformer requested %u", maxConfs)

    dots = oechem.OEDots(10000, 200, "molecules")
    for mol in ifs.GetOEMols():
        if maxConfs is not None:
            TrimConformers(mol, maxConfs)

        oefastrocs.OEPrepareFastROCSMol(mol)
        if not itf.GetBool("-storeFloat"):
            halfMol = oechem.OEMol(mol, oechem.OEMCMolType_HalfFloatCartesian)
            oechem.OEWriteMolecule(ofs, halfMol)
        else:
            oechem.OEWriteMolecule(ofs, mol)

        dots.Update()

    dots.Total()
    ofs.close()

    print("Indexing %s" % itf.GetString("-out"))
    if not oechem.OECreateMolDatabaseIdx(itf.GetString("-out")):
        oechem.OEThrow.Fatal("Failed to index %s" % itf.GetString("-out"))










