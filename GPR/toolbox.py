# -*- coding: utf-8 -*-
import os, glob, re, pickle, subprocess
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import urllib
from openbabel import pybel


def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter




def get_files(path):
    convert = lambda txt: int(txt) if txt.isdigit() else txt
    return sorted(glob.glob(path), key=lambda x:[convert(s) for s in re.split("([0-9]+)",x)])


def mkdir(path):
    """Function will create a directory if given path does not exist.

    >>> toolbox.mkdir("./doctest")
    """
    # create a directory for each system
    if not os.path.exists(path):
        os.makedirs(path)


def save_object(obj, filename):
    """Saves python object as pkl file"""

    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def IUPAC_from_smiles(smiles, representation="IUPAC Name", print_possbile_conversions=False):

    site = "https://cactus.nci.nih.gov/chemical/structure"
    session = requests.Session()
    content = session.get(site).content
    soup = BeautifulSoup(content, "lxml")
    form = soup.find('form')
    fields = soup.find_all("input")
    formdata = dict((field.get('name'), field.get('value')) for field in fields)
    # we need to change the smile string the main field as well as the other fields
    formdata['identifier'] = u"%s"%smiles
    fields = soup.find_all("select")
    fields = list(str(fields).replace("\n","").split("</option>"))
    if print_possbile_conversions:
        print(fields)
        exit()
    for i,field in enumerate(fields):
        if str(field).split(">")[-1] == representation:
            otherformdata = str(fields[i]).split('<option value="')[-1].split('">')[0]
    formdata['representation'] = u"%s"%otherformdata

    site = site+"/"+str(u"%s"%smiles)+"/"+str(formdata['representation'])
    session = requests.Session()
    content = str(session.get(site).content)
    if "Page not found" in content: content=np.nan
    if "DOCTYPE" in content: content=np.nan
    return content



def structure_from_smile(smile, file_format="pdb", filename="structure", verbose=False):
    """Uses BeautifulSoup to get structures from SMILES translator provided by
    National Cancer Institute CADD group (https://cactus.nci.nih.gov/translate/).

    Args:
        smile(str) - smile string
        file_format(str) - any one of {SDF, PDB, MOL}
    """

    site = "https://cactus.nci.nih.gov/translate/"
    session = requests.Session()
    content = session.get(site).content
    soup = BeautifulSoup(content, "lxml")
    #print(soup.prettify())
    form = soup.find('form')
    fields = soup.find_all("input")
    formdata = dict((field.get('name'), field.get('value')) for field in fields)
    # we need to change the smile string the main field as well as the other fields
    formdata['smiles'] = u"%s"%smile
    formdata['format'] = u"%s"%file_format
    if verbose:
        print(formdata)
    posturl = urllib.parse.urljoin(site, form['action'])
    new_content = session.post(posturl, data=formdata)
    if verbose:
        print(new_content.text)
    # now we need to save the file
    soup = BeautifulSoup(new_content.content, "lxml")
    atags = soup.find_all("a") # get all of the a-tags (links)
    # click the link to download
    for tag in atags:
        if "Click" in tag.text:
            link = tag.attrs["href"]
            URL = urllib.parse.urljoin(posturl, link)
            output = requests.get(URL, stream=True, allow_redirects=True)
            out_name = str(filename)#+"."+file_format
            with open(out_name, 'wb') as f:
                for chunk in output:
                    f.write(chunk)
            f.close()
            if verbose:
                print("Wrote %s..."%out_name)


def convert_from_smile(smile, file_format="pdb", filename="structure", verbose=False):
    """
    Uses OpenBabel to get structures from SMILES
    https://openbabel.org/docs/dev/UseTheLibrary/Python_PybelAPI.html

    Args:
        smile(str) - smile string
        file_format(str) - any one of {SDF, PDB, MOL}
    """

    mol = pybel.readstring("smi", smile)
    #print(mol.partialcharge)
    mol.title = filename.split("/")[-1]
    mol.make3D(forcefield='mmff94', steps=50)
    out_name = str(filename)#+"."+file_format
    #print(pybel.outformats)
    file = pybel.Outputfile(file_format, filename=out_name, overwrite=True)
    file.write(mol)


def mol2_from_xyz(xyz, filename="structure", verbose=False):
    """
    Uses OpenBabel to get structures from SMILES
    https://openbabel.org/docs/dev/UseTheLibrary/Python_PybelAPI.html

    Args:
        smile(str) - smile string
        file_format(str) - any one of {SDF, PDB, MOL}
    """

    mol = pybel.readfile("xyz", xyz)
    file = pybel.Outputfile("MOL", filename, overwrite=True)
    file.write(mol)

def mol2_from_PDB(PDB, filename="structure", verbose=False):
    """
    Uses OpenBabel to get structures from SMILES
    https://openbabel.org/docs/dev/UseTheLibrary/Python_PybelAPI.html

    Args:
        smile(str) - smile string
        file_format(str) - any one of {SDF, PDB, MOL}
    """

    subprocess.Popen(f"obabel -ipdb {PDB} -omol2 {filename} > {filename}", shell=True)
    #mol = pybel.readfile("PDB", PDB)
    #file = pybel.Outputfile("MOL", filename, overwrite=True)
    #file.write(mol)

def XYZ_from_PDB(PDB, filename="structure", verbose=False):
    """
    Uses OpenBabel to get structures from SMILES
    https://openbabel.org/docs/dev/UseTheLibrary/Python_PybelAPI.html

    Args:
        smile(str) - smile string
        file_format(str) - any one of {SDF, PDB, MOL}
    """

    mol = pybel.readfile("pdb", PDB)
    file = pybel.Outputfile("xyz", filename, overwrite=True)
    file.write(mol)

def PDB_from_XYZ(XYZ, filename="structure", verbose=False):
    """
    Uses OpenBabel to get structures from SMILES
    https://openbabel.org/docs/dev/UseTheLibrary/Python_PybelAPI.html

    Args:
        smile(str) - smile string
        file_format(str) - any one of {SDF, PDB, MOL}
    """
    subprocess.Popen(f"obabel -ixyz {XYZ} -opdb {filename} > {filename}", shell=True)

    #mol = pybel.readfile("xyz", XYZ)
    #file = pybel.Outputfile("pdb", , overwrite=True)
    #file.write(mol)


def PDB_from_MOL(MOL, filename="structure", verbose=False):
    """
    Uses OpenBabel to get structures from SMILES
    https://openbabel.org/docs/dev/UseTheLibrary/Python_PybelAPI.html

    Args:
        smile(str) - smile string
        file_format(str) - any one of {SDF, PDB, MOL}
    """
    subprocess.Popen(f"obabel -imol {MOL} -opdb {filename} > {filename}", shell=True)
    #mol = pybel.readfile("mol", MOL)
    #file = pybel.Outputfile("pdb", filename, overwrite=True)
    #file.write(mol)









if __name__ == "__main__":

    import doctest
    doctest.testmod()







