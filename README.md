# GP-SAMPL7


**Stacking Gaussian Processes to Improve pKa Predictions in the SAMPL7 Challenge**

**Authors**
* Robert M. Raddi
    - Department of Chemistry, Temple University
* Vincent A. Voelz
    - Department of Chemistry, Temple University
    

Prediction of relative free energies and macroscopic $pK_{a}$s for SAMPL6 and SAMPL7 small molecules using a standard Gaussian process regression as well as a deep Gaussian process regression.


### Below is a description of what this repository contains:

- [`GPR/`](GPR/): the code to perform standard and deep GPR, processing, analysis, etc.
- [`scripts_and_notebooks/`](scripts_and_notebooks/): scripts that call on `GPR/` e.g., script to get features, analysis notebooks
- [`Structures/`](Structures/): input smiles strings, input microtransitions
- [`Submissions/`](Submissions/): SAMPL7 submission (**only for the standard GP model**)
- [`predictions/`](predictions/): directories separating results & prediction files for SAMPL6 and SAMPL7. Each consisting of relative free energies and macroscopic $pK_{a}$ values for each small molecule.
- [`pKaDatabase/`](pKaDatabase/): curated database from various sources.
- [`tables/`](tables/): LaTeX tables for free energies and macroscopic pKas
- [`figures/`](figures/): macroscopic pKa comparison between std GP and deep GP








