# load python libraries:{{{
import sys,os
import GPR
#from GPR import toolbox
from GPR.toolbox import mkdir, get_files
from itertools import combinations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdEHTTools
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
IPythonConsole.ipython_useSVG=False
pd.options.display.max_rows = 25
from IPython.display import display, HTML
import warnings
warnings.filterwarnings("ignore")
pd.options.display.max_columns = 50
pd.options.display.max_rows = 20
#pd.options.display.max_rows = 10
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF,Matern)


from GPR import OE # TODO: remove this for all open source libraries
#:}}}

# Check environment :{{{

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


#:}}}

# Write input files:{{{
def write_input_files(dir):
    GPR.toolbox.mkdir(dir+"input_files/")
    path = dir+"microstates/*_microstates.csv"
    files = GPR.toolbox.get_files(path)
    for file in files:
        SM_ID = file.split("/")[-1].split("_")[0]
        GPR.toolbox.mkdir(dir+"input_files/"+SM_ID)
        print(f"Processing {SM_ID} and writing input file...")
        csv = pd.read_csv(file)
        microstate_smiles = pd.read_csv(dir+"microstates/"+SM_ID+"_microstates.csv")
        df = pd.DataFrame()
        for i,smiles in enumerate(microstate_smiles["canonical isomeric SMILES"]):
            microstate = list(microstate_smiles["microstate ID"])[i]
            #smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
            try: Chem.MolFromSmiles(smiles, sanitize=True)
            except(Exception) as e:
                mol = GPR.RD.fix_charges(smiles, verbose=False)
                smiles = Chem.MolToSmiles(mol)
            bool,df = GPR.RD.append_microstate_to_df(df, smiles, macrostateID=SM_ID, microstateID=int(microstate[-3:]))
        input_file = dir+"input_files/"+SM_ID+"/"+SM_ID+".csv"
        df.to_csv(input_file, index=False)

        # NOTE: Now get the microtransitions for each SM
        microtransitions = GPR.Processing.get_micro_transitions(input_file, column=3, ref="000")
        transition_file = dir+"input_files/"+SM_ID+"/"+SM_ID+"_transitions_dict.pkl"
        # save dictionary object
        GPR.toolbox.save_object(microtransitions, transition_file)
        # convert to indices and save indices as numpy
        transition_file = dir+"input_files/"+SM_ID+"/"+SM_ID+"_transitions.npy"
        indices = []
        for key,val in microtransitions.items():
            microstateIdx = np.where(key == np.array(list(df["microstate ID"])))[0]
            microstateIdx = np.int(microstateIdx)
            valIdx = [int(np.where(value == np.array(list(df["microstate ID"])))[0]) for value in np.array(val)]
            [indices.append([microstateIdx,Idx]) for Idx in valIdx]
        transitions = np.array(indices)
        np.save(transition_file, transitions)
        GPR.Graph.graph_from_df_(input_file, transition_file, column=3,
                molSize=0.20, subImgSize=(150,150), figsize=(6, 10),
                figname=dir+"input_files/"+SM_ID+"/Network_of_states.png")
    #"microstate ID 1", "microstate ID 2"
    #SM01_micro001,SM01_micro005


#write_input_files(dir="Structures/GPR/")
#exit()
#:}}}

# Get_features:{{{
def get_total_features(path):
    """
    """

    files = GPR.toolbox.get_files(path)
    for file in files:
        SM_ID = file.split("/")[-1].split("_")[0].split(".csv")[0]

        #########################################################
        #NOTE: Testing
        #if int(SM_ID.split("SM")[-1]) <= 42: continue
        #########################################################

        print(f"Getting features for {SM_ID}...")
        # Get features from input_file (all transitions)
        df = pd.read_csv(file)
        path_to_input_files = path.split("input_files/")[0]+"input_files/"
        output_path = f"{SM_ID}"
        GPR.toolbox.mkdir(output_path)
        # NOTE: get microtransitions
        #microtransitions = GPR.Processing.get_micro_transitions(input_file)
        path_to_transitions = path_to_input_files+SM_ID+"/"+f"{SM_ID}_transitions.npy"
        microtransitions = np.load(path_to_transitions)
        #microtransitions = GPR.RD.check_transitions(microtransitions, df)
        total_df = GPR.features.get_features(df, microtransitions, verbose=1)
        print(total_df)
        total_df.to_pickle(os.path.join(output_path,"total_feature_input.pkl"))

#get_total_features(path="Structures/GPR/input_files/*/SM*.csv")
#exit()
#:}}}

# Get training data:{{{

#training_set = pd.read_pickle("pKaDatabase/Database_final.pkl")

#training = pd.read_pickle("pKaDatabase/Database_final.pkl")
training = pd.read_pickle("pKaDatabase/Database_final.pkl")

####################################################
#feat = "pKa"
#cond = np.where((-5 <= training[feat]) & (training[feat] <= 6))[0][::2]
#training.drop(index=cond, inplace=True)
#training = training.reset_index(drop=True)
#cond = np.where((-5 <= training[feat]) & (training[feat] <= 6))[0][::3]
#training.drop(index=cond, inplace=True)
####################################################

dataset = pd.read_pickle("pKaDatabase/Sulfonamides.pkl")
training_set = pd.concat([training, dataset])#[::2]



#NOTE: from model val
#file = f"model_validation/mono/0.66/training_set_k_fold_cross_val_1380_mono.pkl"
#kfold_train = pd.read_pickle(file)
#training_set = training_set.iloc[list(kfold_train.index)]

#NOTE: Other databases
#training_set = pd.read_pickle("pKaDatabase/Database.pkl")
#a = pd.read_pickle("pKaDatabase/OChemDatabase_final.pkl")
#b = pd.read_pickle("pKaDatabase/pKaDatabase_final.pkl")
#training_set = pd.concat([a,b], axis=0)
#training_set.to_pickle("pKaDatabase/Database.pkl")



# Curate dataset:{{{
def curate_dataset(dataset, iterations=15):
    for i in range(iterations):
        dataset = GPR.RD.drop_duplicates(dataset, key1='deprotonated microstate smiles',
                key2='protonated microstate smiles', threshold=0.03)
        dataset = dataset.reset_index(drop=True)
    # NOTE: Compute num of ionizable sites and molecular weight
    dataset = GPR.RD.get_num_ionizable_groups(db=dataset, smiles_key="protonated microstate smiles")
    dataset = GPR.RD.get_mol_weight(dataset, smiles_key='protonated microstate smiles')
    return dataset

#training_set = curate_dataset(training_set, iterations=16)
#print(training_set.info())
#training_set.to_pickle("pKaDatabase/Database_final.pkl")
#exit()
#:}}}


feats = [#'deprotonated microstate ID', 'protonated microstate ID',
         'deprotonated microstate smiles', 'protonated microstate smiles',
         'AM1BCC partial charge (prot. atom)',
         'AM1BCC partial charge (deprot. atom)',
         'AM1BCC partial charge (prot. atoms 1 bond away)',
         'AM1BCC partial charge (deprot. atoms 1 bond away)',
         'AM1BCC partial charge (prot. atoms 2 bond away)',
         'AM1BCC partial charge (deprot. atoms 2 bond away)',
         '∆G_solv (kJ/mol) (prot-deprot)',
         'SASA (Shrake)',
         'Bond Order',
         'Change in Enthalpy (kJ/mol) (prot-deprot)', 'pKa',
         'num ionizable groups']

weight = []
for smiles in list(training_set["deprotonated microstate smiles"]):
    weight.append(GPR.RD.get_MolWt_from_smiles(smiles))
training_set["Weight"] = weight
training_set = training_set.iloc[np.where(training_set["Weight"]<=500)]
training = training_set

feat = "num ionizable groups"
#value = (1,2)
#value = (0,1)
#cond0 = np.where((training[feat]==value[0]) | (training[feat]==value[1]))
cond0 = np.where((training[feat]<=2))
#feat = "∆G_solv (kJ/mol) (prot-deprot)"
feat = "pKa"
cond5 = np.where((0 <= training[feat]) & (training[feat] <= 12))
#cond5 = np.where((-5 <= training[feat]) & (training[feat] <= 15))

ts = GPR.RD.get_fingerprints(training, "protonated microstate smiles")
#moi_smiles = list(total_df.iloc[[0]]["protonated microstate smiles"])[0]
#sim = GPR.RD.get_similarity(ts, moi_smiles)
#cond4 = np.where(sim["Fingerprint Similarity"] >= 0.00)
#NOTE:
subset = list(set(cond0[0])&set(cond5[0]))
#subset = list(set(cond5[0]))
training = training.iloc[subset]
training = training[feats[2:-1]]
#:}}}

# Training for kernel parameters:{{{
def train_for_kernel_parameters(file, training, feats):
    training = training[feats[2:-1]]
    training = training.dropna()
    X,y = training.drop("pKa", axis=1).to_numpy(), training['pKa'].to_numpy()

    def function():
        """Testing with an arbitary function..."""
        return feat_vector

    # optimizer:{{{
    def optimizer(obj_func, initial_theta, bounds):
        # * 'obj_func' is the objective function to be minimized, which
        #   takes the hyperparameters theta as parameter and an
        #   optional flag eval_gradient, which determines if the
        #   gradient is returned additionally to the function value
        # * 'initial_theta': the initial value for theta, which can be
        #   used by local optimizers
        # * 'bounds': the bounds on the values of theta
        #
        import scipy
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
        # NOTE: Possible methods:
        # 'Nelder-Mead','Powell','L-BFGS-B','TNC','COBYLA','SLSQP','trust-constr’
        opt_res = scipy.optimize.minimize(obj_func, initial_theta, method='L-BFGS-B',#"trust-constr",
                jac=True, bounds=bounds)
        theta_opt, func_min = opt_res.x, opt_res.fun
        # Returned are the best found hyperparameters theta and
        # the corresponding value of the target function.
        return theta_opt, func_min

    #:}}}

    # plot k-fold:{{{

    def plot_k_fold(file, filename="stats.pdf"):

        fig = plt.figure()
        ax2 = plt.subplot(1,1,1)
        results = pd.read_pickle(file)
        posterior = np.concatenate(list(results["posterior"]))
        y_test = np.concatenate(list(results["y_test"]))
        lower_lim, upper_lim = -4, 15
        limit = np.array([lower_lim, upper_lim])
        ax2.scatter(y_test, posterior, s=15)
        ax2.plot(limit, limit, color="k")
        ax2.fill_between(limit, limit+1.0, limit-1.0, facecolor='brown', alpha=0.35)
        ax2.fill_between(limit, limit+2.0, limit-2.0, facecolor='wheat', alpha=0.50)
        ax2.set_xlabel("Observed", size=16)
        ax2.set_ylabel("Predicted", size=16)
        ax2.set_xlim(limit)
        ax2.set_ylim(limit)
        posx,posy = np.min(y_test)+4.5, np.max(posterior) - 4
        string = r'''$N$ = %i
        $R^{2}$ = %0.3f
        MAE = %0.3f
        MSE = %0.3f
        RMSE = %0.3f'''%(len(posterior), results.mean()["R^2"], results.mean()["MAE"], results.mean()["MSE"], results.mean()["RMSE"])
        ax2.annotate(string, xy=(posx,posy), xytext=(posx,posy))
        #ax2.axvline(x=4, linewidth=4, color='r')
        #ax2.axvline(x=9, linewidth=4, color='r')
        ###########################################
        fig = ax2.get_figure()
        fig.tight_layout()
        fig.savefig(filename)



    #:}}}


    ###############################################################################
    # NOTE: greater length scale for hgiher variation within a feature
    ###############################################################################
    length_scale = [1.,1.,1.,1.,1.,1.,1.,1.,1.,1.]
    kernel =  1.*Matern(length_scale=length_scale, length_scale_bounds=(1e-1, 1e3), nu=2.5)
    ###############################################################################
    # Error standard deviation.
    sigma_n = 1e-6 # by default 1e-5
    alpha=sigma_n**2
    normalize_y = 1
    random_state = 1
    copy_X_train = 1
    n_restarts_optimizer=50
    #rounds_of_opt = 5
    optimizer = optimizer
    ###############################################################################
    gp = GaussianProcessRegressor(kernel, alpha, optimizer, n_restarts_optimizer,
            normalize_y, copy_X_train, random_state)

    # Fit the GP model to the data performing maximum likelihood estimation
    #gp.fit(X, y)

    # Deactivate maximum likelihood estimation for the cross-validation loop
    #gp.theta0 = gp.kernel_.theta  # Given correlation parameter = MLE
    #gp.thetaL, gp.thetaU = None, None  # None bounds deactivate MLE
    print("Running k fold cross validation...")
    results = GPR.statistics.k_fold_cross_val(gp, X, y, folds=3)
    results.to_pickle(file)
    results = pd.read_pickle(file)
    plot_k_fold(file, filename="k_fold_stats.pdf")

training = training.dropna()
#training = training.iloc[::3]
#training = pd.concat([training.iloc[0:1000],training.iloc[1600:2505]])
num = len(list(training["pKa"]))
print(num)
#exit()

#file = f"training_set_k_fold_cross_val_{num}_final.pkl"
#training.to_pickle(file)
#file = f"k_fold_cross_val_results_{num}_final.pkl"
#train_for_kernel_parameters(file, training, feats)
#exit()
file = "k_fold_cross_val_results_1122_final.pkl"


#file = f"model_validation/mono/0.66/k_fold_cross_val_results_norm_ochem_final_1380_mono.pkl"
results = pd.read_pickle(file)
#theta = results.iloc[:]["theta"].mean()
kernel = list(results["kernel"])[0]
kernels = {"Matern": kernel}

training_x, training_y = training.drop("pKa", axis=1).to_numpy(), training['pKa'].to_numpy()

#:}}}

# GPR:{{{
path = "SM*/total_feature_input.pkl"
files = GPR.toolbox.get_files(path)
for i,file in enumerate(files):
    dir = "Structures/SAMPL6/"
    SM_ID = file.split("/")[0]
    print(f"Molecule of interest: {SM_ID}")
    output_path = f"{SM_ID}"
    GPR.toolbox.mkdir(output_path)
    total_df = pd.read_pickle(file)

    feature_vector = total_df[feats[2:-2]]
    #feature_vector = feature_vector.dropna()
    feature_vector.to_pickle(os.path.join(output_path,"feature_vector.pkl"))
    feat_vector = pd.read_pickle(os.path.join(output_path,"feature_vector.pkl")) #= np.load("feature_vector.npy", allow_pickle=True)
    feature_list = list(feat_vector.keys())
    feature_list.append("pKa")

    print("Running GPR ...")

    def function():
        """Testing with an arbitary function..."""
        return feat_vector.to_numpy()


    def function_with_noise():
        """Function to create noise e.g., Gaussian white noise
        for the GaussianProcessRegressor as the parameter alpha.
        """

        y = function()
        # Observations and noise
        for i in range(len(y[0])):
            dy = np.random.random(y[:,i].shape)
            noise = np.random.normal(np.mean(y[:,i]), dy)
            y[:,i] += noise
        return y

    # optimizer:{{{
    def optimizer(obj_func, initial_theta, bounds):
        # * 'obj_func' is the objective function to be minimized, which
        #   takes the hyperparameters theta as parameter and an
        #   optional flag eval_gradient, which determines if the
        #   gradient is returned additionally to the function value
        # * 'initial_theta': the initial value for theta, which can be
        #   used by local optimizers
        # * 'bounds': the bounds on the values of theta
        #
        import scipy
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
        # NOTE: Possible methods:
        # 'Nelder-Mead','Powell','L-BFGS-B','TNC','COBYLA','SLSQP','trust-constr’
        opt_res = scipy.optimize.minimize(obj_func, initial_theta, method='L-BFGS-B',#"trust-constr",
                jac=True, bounds=bounds)
        theta_opt, func_min = opt_res.x, opt_res.fun
        # Returned are the best found hyperparameters theta and
        # the corresponding value of the target function.
        return theta_opt, func_min

    #:}}}

    ###############################################################################
    # NOTE: greater length scale for hgiher variation within a feature
    ###############################################################################
    length_scale = [1.,1.,1.,1.,1.,1.,1.,1.,1.,1.]

    # Setting the domain
    num = len(list(total_df["deprotonated microstate smiles"]))
    domain = dict(start=0, stop=num, num=num)
    df = {"x": np.linspace(domain["start"], domain["stop"], domain["num"])}

    ###############################################################################
    # Error standard deviation.
    sigma_n = 1e-6 # by default
    alpha=sigma_n**2
    noise=0
    normalize_y = 1
    random_state=1
    copy_X_train=1
    nsamples=len(training_x)
    n_restarts_optimizer=5#30
    #rounds_of_opt = 5
    optimizer=None
    #optimizer = optimizer
    ###############################################################################
    sample_parameters = dict(min=np.min(training_x), max=np.max(training_x), nsamples=nsamples, distribution=None, function=function,)
    pd.options.display.max_rows = 30
    for name,kernel in kernels.items():
        # Specify Gaussian Process
        #-------------------------
        # n_restarts_optimizer= The number of restarts of the optimizer for finding the kernel’s
        # parameters which maximize the log-marginal likelihood.

        #kernel = kernel.clone_with_theta(theta)
        #print(kernel)

        gp = GaussianProcessRegressor(kernel, alpha, optimizer, n_restarts_optimizer,
                normalize_y, copy_X_train, random_state)


        if noise==True: sample_parameters["function"]=function_with_noise

        df,fig = GPR.GPR.plot_posterior(df, gp, sample_parameters, training_x, training_y,
            feature_list=['PC (AH)', 'PC (A)', 'PC (AH 1bond)', 'PC (A 1bond)',
                'PC (AH 2bond)', 'PC (A 2bond)', r'$∆G_{solv}$(AH-A) (kJ/mol) ',
                'SASA', 'Bond Order', '∆H(AH-A) (kJ/mol)', 'pKa'],
            random_state=random_state, title=False, output_path=output_path)
        #print(test)
        fig.savefig(os.path.join(output_path,f"{name}_Posterior_only.png"))


        df.to_pickle(os.path.join(output_path,f"results.pkl"))
        print(f"Kernel: {name}")
        if isnotebook():
            display(HTML(pd.DataFrame(kernel.get_params()).to_html()))
            display(HTML(df.head().to_html()))
        else:
            print(kernel.get_params())
            print(df)
        print("\n\n")

    __doc__='''
    Description of Plots
    --------------------
                                               ( df = {"x": np.linspace(%s, %s, %s)}                  )
    Actual curve: black dotted line            ( df["Actual"] = function(df["x"])                     )
    Sampled points: Red dots                   ( sample_parameters = dict(min=%s, max=%s, nsamples=%s )
    Kernel: Colorful squiggles
    Prediciton: Solid black line
    Prior distribution: Plots on left
    Posterior distribution: Plots on right
    '''%(domain["start"], domain["stop"], domain["num"],
         sample_parameters["min"],sample_parameters["max"],sample_parameters["nsamples"])
    #print(f"\n{__doc__}\n")

    fig1 = GPR.GPR.plot_n_dimensions(sample_parameters, training_x, training_y, feature_list)
    fig1.savefig(os.path.join(output_path,'features.png'))


    #:}}}


