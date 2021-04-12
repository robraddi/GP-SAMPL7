# GP-SAMPL7


**Stacking Gaussian Processes to Improve pKa Predictions in the SAMPL7 Challenge**

**Authors**
* Robert M. Raddi
    - Department of Chemistry, Temple University
* Vincent A. Voelz
    - Department of Chemistry, Temple University
    

Prediction of relative free energies and macroscopic pKas for SAMPL6 and SAMPL7 small molecules using a standard Gaussian process regression as well as a deep Gaussian process regression.


### Below is a description of what this repository contains:

- [`GPR/`](GPR/): the code to perform standard and deep GPR, processing, analysis, etc.
- [`scripts_and_notebooks/`](scripts_and_notebooks/): scripts that call on `GPR/` e.g., script to get features, analysis notebooks
  - [`scripts_and_notebooks/compile_results.ipynb`](scripts_and_notebooks/compile_results.ipynb): **master notebook for creating figures and tables**
  - [`scripts_and_notebooks/database_info.ipynb`](scripts_and_notebooks/database_info.ipynb): notebook for analyzing the database
  - [`scripts_and_notebooks/features.py`](scripts_and_notebooks/features.py): script for computing descriptors
  - [`scripts_and_notebooks/standardGPR.py`](scripts_and_notebooks/standardGPR.py): script for running `sklearn.GaussianProcessRegressor`
  - [`scripts_and_notebooks/runme_deepGP.py`](scripts_and_notebooks/runme_deepGP.py): script for running deep Gaussian process regression using `deepGPy`
- [`Structures/`](Structures/): input smiles strings, input microtransitions
- [`Submissions/`](Submissions/): SAMPL7 submission (**only for the standard GP model**)
- [`predictions/`](predictions/): directories separating results & prediction files for SAMPL6 and SAMPL7. Each consisting of relative free energies and macroscopic pKa values for each small molecule.
  - [`predictions/SAMPL6_deepGP/`](predictions/SAMPL6_deepGP/): free energies and macro-pKa predictions
  - [`predictions/SAMPL6_stdGP/`](predictions/SAMPL6_stdGP/): free energies and macro-pKa predictions
  - [`predictions/SAMPL7_deepGP/`](predictions/SAMPL7_deepGP/): free energies and macro-pKa predictions
  - [`predictions/SAMPL7_stdGP/`](predictions/SAMPL7_stdGP/): free energies and macro-pKa predictions
- [`pKaDatabase/`](pKaDatabase/): curated database from various sources.
  - [`pKaDatabase/pKaDatabase.pkl`](pKaDatabase/pKaDatabase.pkl): pickle file of database (loads with pandas) **NOTE: feature calculations already performed and stored inside `pd.read_pickle('pKaDatabase.pkl')`**
  - [`pKaDatabase/Sulfonamides.pkl`](pKaDatabase/Sulfonamides.pkl): pickle file of only sulfonamides database (loads with pandas)
- [`tables/`](tables/): LaTeX tables for free energies and macroscopic pKas
- [`figures/`](figures/): macroscopic pKa comparison between std GP and deep GP








