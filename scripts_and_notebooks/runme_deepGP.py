# load python libraries:{{{
import sys,os
import GPR
#from GPR import toolbox
from GPR.toolbox import mkdir, get_files
import GPy
import deepgp
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
from GPR.decorators import multiprocess
from GPR.decorators import Manager
import h5py


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

# Methods:{{{
def decompose_training(training):
    X,y = training.drop(["pKa","Fingerprint"], axis=1).to_numpy(), training['pKa'].to_numpy()
    fingerprints = training["Fingerprint"].to_numpy()
    train = []
    for i,fp in enumerate(fingerprints):
        x = np.concatenate([X[i],fp])
        train.append(x)
    X = np.array(train)
    return X,y

def decompose_testing(testing):
    X,fingerprints = testing.drop(["Fingerprint"], axis=1).to_numpy(), testing["Fingerprint"].to_numpy()
    train = []
    for i,fp in enumerate(fingerprints):
        x = np.concatenate([X[i],fp])
        train.append(x)
    X = np.array(train)
    return X


def visualize_DGP(model, labels=None, layer=0, dims=[0,1]):
    """
    A small utility to visualize the latent space of a DGP.
    """
    import matplotlib.pyplot as plt

    colors = ['r','g', 'b', 'm']
    markers = ['x','o','+', 'v']
    if labels != None:
        for i in range(model.layers[layer].X.mean.shape[0]):
            plt.scatter(model.layers[layer].X.mean[i,0],model.layers[layer].X.mean[i,1],color=colors[labels[i]], s=16, marker=markers[labels[i]])
    else:
        for i in range(model.layers[layer].X.mean.shape[0]):
            plt.scatter(model.layers[layer].X.mean[i,0],model.layers[layer].X.mean[i,1], s=16)





#:}}}

# Get training data:{{{

training = pd.read_pickle("pKaDatabase/pkaDatabase.pkl")

####################################################
#feat = "pKa"
#cond = np.where((-5 <= training[feat]) & (training[feat] <= 6))[0][::2]
#training.drop(index=cond, inplace=True)
#training = training.reset_index(drop=True)
#cond = np.where((-5 <= training[feat]) & (training[feat] <= 6))[0][::3]
#training.drop(index=cond, inplace=True)
####################################################
dataset = pd.read_pickle("pKaDatabase/Sulfonamides.pkl")
training_set = pd.concat([training, dataset])



#NOTE: Subset of training
#training_set = training_set[::10]



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


training_set = GPR.RD.get_fingerprints(training_set, "protonated microstate smiles", astype=np.array)
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
         'Change in Enthalpy (kJ/mol) (prot-deprot)',
         'Fingerprint',
         'pKa',
         'num ionizable groups',
         ]

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
cond0 = np.where((training[feat]<=4))
#feat = "∆G_solv (kJ/mol) (prot-deprot)"
feat = "pKa"
#cond5 = np.where((0 <= training[feat]) & (training[feat] <= 12))
cond5 = np.where((-5 <= training[feat]) & (training[feat] <= 15))

feat = "∆G_solv (kJ/mol) (prot-deprot)"
cond6 = np.where((training[feat] > -4000))
feat = 'Change in Enthalpy (kJ/mol) (prot-deprot)'
cond7 = np.where((training[feat] > -0.8))

#feat = 'Fingerprint'
#cond0 = np.where((training[feat]<=2))

path = "SM*/total_feature_input.pkl"
files = GPR.toolbox.get_files(path)
total_df = pd.read_pickle(files[0])
#total_df = GPR.RD.get_fingerprints(total_df, "protonated microstate smiles", astype=np.array)


path = "Structures/GPR/input_files/SM*/SM*.csv"
files = GPR.toolbox.get_files(path)
ts = GPR.RD.get_fingerprints(training, 'protonated microstate smiles')

avg = []
for file in files:
    input_file = pd.read_csv(file)
    moi_smiles = list(input_file["smiles"])[0]
    sim = GPR.RD.get_similarity(ts, moi_smiles)
    avg.append(sim["Fingerprint Similarity"].mean())
avg = np.average(avg)

total = []
for file in files:
    input_file = pd.read_csv(file)
    moi_smiles = list(input_file["smiles"])[0]
    sim = GPR.RD.get_similarity(ts, moi_smiles)
    cond4 = np.where(sim["Fingerprint Similarity"] >= avg)[0]
    #cond4 = np.where(sim["Fingerprint Similarity"] >= .20)[0]
    total.append(cond4)



total = np.concatenate(total)
cond4 = np.array(list(set(total.tolist())))


#NOTE:
subset = list(set(cond0[0])&set(cond5[0])&set(cond6[0])&set(cond7[0])&set(cond4))

#print(len(subset))
#exit()
#subset = list(set(cond5[0]))
training = training.iloc[subset]
training = training[feats[2:-1]]

#:}}}

# k_fold_cross_val{{{
def k_fold_cross_val(estimator, X, y, scale, offset, folds=3, output_filename="model.hdf5"):
    """scikit-learn k-fold cross-validation"""
    # Perform a cross-validation estimate of the coefficient of determination using
    # the cross_validation module using all CPUs available on the machine

    from sklearn.model_selection import cross_val_score, KFold
    from sklearn import metrics
    with Manager() as manager:
        results = manager.list()
        cv = KFold(folds, True, 1)
        list_of_dict = [{"fold":f, "indices":i} for f,i in enumerate(cv.split(X))]
        @multiprocess(iterable=list_of_dict)
        def pipeline(list_of_dict):
            fold, indices = list_of_dict["fold"], list_of_dict["indices"]
            (train_index, test_index) = indices
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            posterior, posterior_var = estimator.predict(X_test)
            posterior= posterior*scale+offset
            posterior_var *= scale*scale
            posterior_std = np.sqrt(posterior_var)
            posterior = posterior[:,0]
            log_likelihood = estimator.log_likelihood()
            #output_filename = output_filename.replace(".f"_{fold}.hdf5"
            results.append({"fold": fold,
                            "train index": train_index,
                            "test index": test_index,
                            "training size": len(X_test),
                            "posterior": posterior,
                            "posterior_std": posterior_std,
                            "y_test": y_test,
                            "model": estimator,
                            "log likelihood": log_likelihood,
                            "R^2": np.corrcoef(y_test, posterior)[0][1]**2,
                            "MAE": metrics.mean_absolute_error(y_test,  posterior),
                            "MSE": metrics.mean_squared_error(y_test,  posterior),
                            "RMSE": np.sqrt(metrics.mean_squared_error(y_test,  posterior)),
                            "filename": output_filename,
                            })
            #estimator.save_model(output_filename)
            #estimator.save(output_filename)
        return pd.DataFrame(results[:])
# }}}

# Training for kernel parameters:{{{
def train_for_kernel_parameters(file, training, feats):
    training = training[feats[2:-1]]
    training = training.dropna()
    #X,y = training.drop("pKa", axis=1).to_numpy(), training['pKa'].to_numpy()
    X,y = decompose_training(training)

    def function():
        """Testing with an arbitary function..."""
        return feat_vector

    # plot k-fold:{{{

    def plot_k_fold(file, filename="stats.pdf", figsize=(10,10)):
        import scipy
        fig = plt.figure()
        fig.set_figheight(figsize[0]);fig.set_figwidth(figsize[1])
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
        posx,posy = np.min(y_test)+1.25, np.max(posterior) - 4
        string = r'''
        $N$ = %i
        $R^{2}$ = %0.3f
        MAE = %0.3f
        MSE = %0.3f
        RMSE = %0.3f
        SEM = %0.3f'''%(len(posterior), results.mean()["R^2"], results.mean()["MAE"], results.mean()["MSE"], results.mean()["RMSE"],
                       scipy.stats.sem(posterior))
        ax2.annotate(string, xy=(posx,posy), xytext=(posx,posy))
        #ax2.axvline(x=4, linewidth=4, color='r')
        #ax2.axvline(x=9, linewidth=4, color='r')
        ###########################################
        fig = ax2.get_figure()
        fig.tight_layout()
        fig.savefig(filename)



    #:}}}


    # Get mean and Scaling values
    offset = y.mean()
    scale = np.sqrt(y.var())
    #xlim = (np.min(X)-2, np.max(X)+2)
    #ylim = (np.min(y)-2, np.max(y)+2)
    yhat = (y-offset)/scale

    hidden = 5 # number of hidden layers

    #print(X.shape[1])
    #print(y[:,np.newaxis].shape[1])
    layers = [y[:,np.newaxis].shape[1], hidden, X.shape[1]]
    inits = ['PCA']*(len(layers)-1) # NOTE: for unsupervised learning only
    #print(inits)
    #exit()
#    kernels = []
#    for i in layers[1:]:
#        #kernels.append(GPy.kern.Matern32(hidden, ARD=True) + GPy.kern.Bias(i))
#        kernels.append(GPy.kern.Matern32(i, ARD=False))

    kernels = [
            GPy.kern.Matern32(hidden, ARD=True) + GPy.kern.Bias(hidden),
            GPy.kern.Matern32(hidden, ARD=False) + GPy.kern.Bias(X.shape[1])

            #GPy.kern.Matern52(hidden, ARD=True) + GPy.kern.Bias(hidden),
            #GPy.kern.Matern52(hidden, ARD=False) + GPy.kern.Bias(X.shape[1])

            #GPy.kern.Matern32(hidden, ARD=True) + GPy.kern.Matern52(hidden, ARD=True) + GPy.kern.RBF(hidden, ARD=True) + GPy.kern.Bias(hidden),
            #GPy.kern.Matern32(hidden, ARD=False) + GPy.kern.Matern52(hidden, ARD=False) + GPy.kern.RBF(X.shape[1], ARD=False) + GPy.kern.Bias(X.shape[1])
            ]

    n_restarts_optimizer=1
    optimizer = 'bfgs'

    # NOTE: Number of inducing points per layer (can be set to different if given as list).
    # At each layer we use num_inducing points for the variational approximation.
    num_inducing = 200
    gp = deepgp.DeepGP(layers, Y=yhat[:,np.newaxis], X=X, inits=inits,
                      kernels=kernels, num_inducing=num_inducing, back_constraint=False)

#    gp = GPy.models.GPRegression(X, yhat[:,np.newaxis], kernel=kernels[0])


    # NOTE:
    # early training of the model with model parameters constrained. This gives
    #the variational inducing parameters some scope to tighten the bound for the
    # case where the noise variance is small and the variances of the Gaussian
    # processes are around 1.
    # Make sure initial noise variance gives a reasonable signal to noise ratio.
    # Fix to that value for a few iterations to avoid early local minima
    for i in range(len(gp.layers)):
        gp.layers[i].likelihood.variance.constrain_positive(warning=False)
        output_var = gp.layers[i].Y.var() if i==0 else gp.layers[i].Y.mean.var()
        gp.layers[i].Gaussian_noise.variance = output_var*0.01
        gp.layers[i].Gaussian_noise.variance.fix()
        #gp.layers[i].inducing_inputs.constrain_fixed()

    gp.optimize(optimizer, messages=True, max_iters=100)
#    gp.optimize(optimizer, messages=True, max_iters=500)
    #gp.optimize_restarts(num_restarts=10, robust=True, parallel=True, num_processes=6)
    #gp.optimize_restarts(num_restarts=10)

    # Unfix noise variance now that we have initialized the model
    for i in range(len(gp.layers)):
        gp.layers[i].Gaussian_noise.variance.unfix()

    #gp.optimize(optimizer, messages=True, max_iters=500)
    gp.optimize(optimizer, messages=True, max_iters=1500)

    log_likelihood = gp.log_likelihood()
    print(f"Log likelihood: {log_likelihood}")

    print("Running k fold cross validation...")
    results = k_fold_cross_val(gp, X, y, scale, offset, folds=3)
    results.to_pickle(file)
    results = pd.read_pickle(file)
    #NOTE: uncomment
    plot_k_fold(file, filename="k_fold_stats.pdf", figsize=(8,8))

    # From the plots above, we see which ones are the dominant dimensions for each layer.
    # So we use these dimensions in the visualization of the latent space below.
    plt.figure(figsize=(5,5))
    visualize_DGP(gp, labels=None, layer=0, dims=[1,2]); plt.title('Layer 0')
    plt.savefig("visualize_DGP_layer0.png")
#    plt.figure(figsize=(5,5))
#    visualize_DGP(gp, labels=None, layer=1, dims=[0,1]); plt.title('Layer 1')
#    plt.savefig("visualize_DGP_layer1.png")

    # TODO: train for kernel parameters use 4 fold
    # sets are: 1056, 2112, 3200 ( do first two sets frist, while they are running find
    # sulfonamides to add to dataset)
    # TODO: save each subset of the training set along with the results file

#:}}}

# Model selection:{{{
def model_selection(X, y, layers=[], inducing_vars=[]):
    """
    Args
        X - training x
        y - training y
    Returns
        pd.DataFrame
    """

    #from sklearn.model_selection import cross_val_score, KFold
    from sklearn import metrics
    from itertools import product

    output_dir = "models"
    GPR.toolbox.mkdir(output_dir)

    with Manager() as manager:
        results = manager.list()
        # Get mean and Scaling values
        offset = y.mean()
        scale = np.sqrt(y.var())
        yhat = (y-offset)/scale
        #print(X.shape)
        list_of_dict = [{"layers":l, "inducing":i} for l,i in list(product(layers,inducing_vars))]
        #print(list_of_dict)
        #exit()
        @multiprocess(iterable=list_of_dict)
        def pipeline(list_of_dict):
            layers,num_inducing = list_of_dict["layers"], list_of_dict["inducing"]
            filename = f"{layers}_layers__{num_inducing}_inducing"
            hidden = int(layers-1) # number of hidden layers
            layers = [y[:,np.newaxis].shape[1], hidden, X.shape[1]]
            inits = ['PCA']*(len(layers)-1) # NOTE: for unsupervised learning only
            n_restarts_optimizer=1
            optimizer = 'bfgs'
            # TODO: can this be outside the function?
            k1 = GPy.kern.Matern52(hidden, ARD=True)*GPy.kern.Exponential(hidden)
            k2 = GPy.kern.Matern52(hidden, ARD=True)*GPy.kern.White(hidden)
            k3 = GPy.kern.Matern52(hidden, ARD=True)*GPy.kern.Bias(hidden)
            kernel_gpml = k1 + k2 + k3

            kernels = [
                    GPy.kern.RBF(hidden, ARD=True),# + GPy.kern.Bias(hidden),
                    GPy.kern.RBF(input_dim=hidden, ARD=True) + GPy.kern.Bias(X.shape[1])

                    #GPy.kern.Matern32(hidden, ARD=True) + GPy.kern.Bias(hidden),
                    #GPy.kern.Matern32(hidden, ARD=False) + GPy.kern.Bias(X.shape[1])

                    #kernel_gpml,
                    #GPy.kern.Matern52(hidden, ARD=True) + GPy.kern.Bias(hidden),
                    #GPy.kern.Matern52(hidden, ARD=False) + GPy.kern.Bias(X.shape[1])

                    #GPy.kern.Matern32(hidden, ARD=True) + GPy.kern.Matern52(hidden, ARD=True) + GPy.kern.RBF(hidden, ARD=True) + GPy.kern.Bias(hidden),
                    #GPy.kern.Matern32(hidden, ARD=False) + GPy.kern.Matern52(hidden, ARD=False) + GPy.kern.RBF(X.shape[1], ARD=False) + GPy.kern.Bias(X.shape[1])
                    ]

            # NOTE: Number of inducing points per layer (can be set to different if given as list).
            # At each layer we use num_inducing points for the variational approximation.
            gp = deepgp.DeepGP(layers, Y=yhat[:,np.newaxis], X=X, inits=inits,
                              kernels=kernels, num_inducing=num_inducing, back_constraint=False)
            # NOTE:
            # early training of the model with model parameters constrained. This gives
            #the variational inducing parameters some scope to tighten the bound for the
            # case where the noise variance is small and the variances of the Gaussian
            # processes are around 1.
            # Make sure initial noise variance gives a reasonable signal to noise ratio.
            # Fix to that value for a few iterations to avoid early local minima
            for i in range(len(gp.layers)):
                gp.layers[i].likelihood.variance.constrain_positive(warning=False)
                output_var = gp.layers[i].Y.var() if i==0 else gp.layers[i].Y.mean.var()
                gp.layers[i].Gaussian_noise.variance = output_var*0.01
                gp.layers[i].Gaussian_noise.variance.fix()

            #gp.optimize(optimizer, messages=False, max_iters=50)
            gp.optimize(optimizer, messages=False, max_iters=500)

            # Unfix noise variance now that we have initialized the model
            for i in range(len(gp.layers)):
                gp.layers[i].Gaussian_noise.variance.unfix()

            #gp.optimize(optimizer, messages=False, max_iters=50)
            gp.optimize(optimizer, messages=False, max_iters=1500)

            model_results = k_fold_cross_val(gp, X, y, scale, offset, folds=3)
            output_filename = f"{output_dir}/{filename}.pkl"
            model_results.to_pickle(output_filename)
            #estimator.save_model(output_filename)
            output_filename = output_filename.replace(".pkl",".hdf5")
            gp.save(output_filename)
            results.append(model_results)
        return pd.DataFrame(results[:])
#:}}}

training = training.dropna()
#training = training.iloc[::3]
#training = pd.concat([training.iloc[0:1000],training.iloc[1600:2505]])
num = len(list(training["pKa"]))
print(num)
#exit()


model = "deepGP" # "StdGP"

file = f"{model}/training_set_k_fold_cross_val_{num}_final.pkl"
training.to_pickle(file)
file = f"{model}/k_fold_cross_val_results_{num}_final.pkl"
#train_for_kernel_parameters(file, training, feats)
#exit()
training = training[feats[2:-1]]
training = training.dropna()
#X,y = training.drop("pKa", axis=1).to_numpy(), training['pKa'].to_numpy()
X,y = decompose_training(training)
#model_selection(X, y, layers=[6,8,10], inducing_vars=[100,150,200])
#model_selection(X, y, layers=[6], inducing_vars=[150,200])
#exit()

#file = "k_fold_cross_val_results_1122_final.pkl"

#"""
# NOTE: find the model with the smallest error
#file = f"{model}/model_0.hdf5"
file = f"models/6_layers__150_inducing.pkl"
#file = f"models/6_layers__200_inducing.pkl"
f = pd.read_pickle(file)
f = list(f["model"])[0]
#file = f"models/6_layers__150_inducing.hdf5"
#file = f"models/6_layers__200_inducing.hdf5"
#f = h5py.File(file, 'r')
#"""
# NOTE: Once you have the optmized model, then you can load in the paramter array into the model
# You need to alter the code below to work with GPy

training_x, training_y = training.drop("pKa", axis=1).to_numpy(), training['pKa'].to_numpy()
training_x,training_y = decompose_training(training)

#print(training_x)
#exit()



# plot_posterior:{{{
def plot_posterior(df, Regressor, sample_parameters, training_x, training_y,
        feature_list, random_state=0, title=False, output_path="./"):
    """Returns the prior and posterior distributions

    Args:
        df(dict): e.g. {"min": 0, "max":10, "num": 1000}
        Regressor(object): Scikitlearn Regressor object
        sample_parameters(dict): dictionary of kwargs

    Returns:
        df(DataFrame): pandas dataframe of prior, posterior, etc.
        fig(figure): dataframe figure
    """

    # TODO: find a way to make this more general and extensible (maybe change x to something else?)
    x = df["x"]
    # Get Figure
    fig, (ax2) = plt.subplots(1)
    fig.set_figheight(4);fig.set_figwidth(10)

    para = sample_parameters
    x = df["x"]
    y = para["function"]()
    posterior, posterior_var = Regressor.predict(y)
    posterior= posterior*scale+offset
    posterior_var *= scale*scale
    posterior_std = np.sqrt(posterior_var)[:,0]
    posterior = posterior[:,0]
    df["posterior"], df["posterior_std"] = posterior, posterior_std
    #plot_kernel(X=x, y=y_samples, Σ=posterior_cov, description="kernel",
    #        xlim=(min(x), max(x)), scatter=False, rotate_x_labels=False,
    #        figname="kernel_plot_posterior.pdf", output_path=output_path)
    # NOTE: this indented comments are for test
    data = pd.concat([pd.DataFrame(y), pd.DataFrame(df["posterior"])], axis=1)
    data = data.reindex()
    data.columns = feature_list
    #training_data = pd.concat([pd.DataFrame(training_x), pd.DataFrame(training_y)], axis=1)
    #training_data = training_data.reindex()
    #training_data.columns = feature_list
    ax2.fill_between(x, df["posterior"] - df["posterior_std"],
            df["posterior"] + df["posterior_std"], alpha=0.3, color='k')
    ax2.plot(x, df["posterior"], "k", lw=1)
    ax2.set_xlim(np.min(x), np.max(x))
    fig.tight_layout()#pad=0.4, w_pad=0.5, h_pad=1.0)
    if title:
        fig.title(title)
    df = pd.DataFrame(df)
    fig.show()
    return df, fig

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
    #print(total_df)
    total_df = GPR.RD.get_fingerprints(total_df, "protonated microstate smiles", astype=np.array)
    #print(total_df)
    #exit()

    feature_vector = total_df[feats[2:-2]]
    #feature_vector = feature_vector.dropna()
    feature_vector.to_pickle(os.path.join(output_path,"feature_vector.pkl"))
    feat_vector = pd.read_pickle(os.path.join(output_path,"feature_vector.pkl")) #= np.load("feature_vector.npy", allow_pickle=True)
    feature_list = list(feat_vector.keys())
    feature_list.append("pKa")

    print("Running GPR ...")

    def function():
        """Testing with an arbitary function..."""

        return decompose_testing(testing=feat_vector)


    ###############################################################################
    # NOTE: greater length scale for hgiher variation within a feature
    ###############################################################################
    length_scale = [1.,1.,1.,1.,1.,1.,1.,1.,1.,1.]

    # Setting the domain
    num = len(list(total_df["deprotonated microstate smiles"]))
    domain = dict(start=0, stop=num, num=num)
    df = {"x": np.linspace(domain["start"], domain["stop"], domain["num"])}

    nsamples=len(training_x)
    sample_parameters = dict(min=np.min(training_x), max=np.max(training_x), nsamples=nsamples, distribution=None, function=function,)
    pd.options.display.max_rows = 30

    # Get mean and Scaling values
    #training_x, training_y
    offset = training_y.mean()
    scale = np.sqrt(training_y.var())
    xlim = (np.min(training_x)-2, np.max(training_x)+2)
    ylim = (np.min(training_y)-2, np.max(training_y)+2)
    yhat = (training_y-offset)/scale

    hidden = 5 # number of hidden layers
    layers = [training_y[:,np.newaxis].shape[1], hidden, training_x.shape[1]]
    inits = ['PCA']*(len(layers)-1)
#    kernels = []
#    for i in layers[1:]:
#        kernels.append(GPy.kern.Matern32(i, ARD=True))

    k1 = GPy.kern.Matern52(hidden, ARD=True)*GPy.kern.Exponential(hidden)
    k2 = GPy.kern.Matern52(hidden, ARD=True)*GPy.kern.White(hidden)
    k3 = GPy.kern.Matern52(hidden, ARD=True)*GPy.kern.Bias(hidden)
    kernel_gpml = k1 + k2 + k3

    kernels = [
            GPy.kern.RBF(hidden, ARD=True),# + GPy.kern.Bias(hidden),
            GPy.kern.RBF(input_dim=hidden, ARD=True) + GPy.kern.Bias(X.shape[1])

            #GPy.kern.Matern32(hidden, ARD=True) + GPy.kern.Bias(hidden),
            #GPy.kern.Matern32(hidden, ARD=False) + GPy.kern.Bias(X.shape[1])

            #kernel_gpml,
            #GPy.kern.Matern52(hidden, ARD=True) + GPy.kern.Bias(hidden),
            #GPy.kern.Matern52(hidden, ARD=False) + GPy.kern.Bias(X.shape[1])

            #GPy.kern.Matern32(hidden, ARD=True) + GPy.kern.Matern52(hidden, ARD=True) + GPy.kern.RBF(hidden, ARD=True) + GPy.kern.Bias(hidden),
            #GPy.kern.Matern32(hidden, ARD=False) + GPy.kern.Matern52(hidden, ARD=False) + GPy.kern.RBF(X.shape[1], ARD=False) + GPy.kern.Bias(X.shape[1])
                    ]

    n_restarts_optimizer=1
    optimizer = 'bfgs'
    gp = deepgp.DeepGP(layers, Y=yhat[:,np.newaxis], X=training_x, inits=inits,
                      kernels=kernels, num_inducing=150, back_constraint=False)
                      #kernels=kernels, num_inducing=200, back_constraint=False)

#    gp = GPy.models.GPRegression(training_x, yhat[:,np.newaxis], kernel=kernels[0])
    gp[:] = f.param_array


    #df,fig = plot_posterior(df, gp, sample_parameters, training_x, training_y,
    #    feature_list=['PC (AH)', 'PC (A)', 'PC (AH 1bond)', 'PC (A 1bond)',
    #        'PC (AH 2bond)', 'PC (A 2bond)', r'$∆G_{solv}$(AH-A) (kJ/mol) ',
    #        'SASA', 'Bond Order', '∆H(AH-A) (kJ/mol)', 'pKa'],
    #    title=False, output_path=output_path)

    y = sample_parameters["function"]()
    posterior, posterior_var = gp.predict(y)
    posterior= posterior*scale+offset
    posterior_var *= scale*scale
    posterior_std = np.sqrt(posterior_var)[:,0]
    posterior = posterior[:,0]
    df = pd.DataFrame()
    df["posterior"], df["posterior_std"] = posterior, posterior_std

    #print(test)
    #fig.savefig(os.path.join(output_path,f"{name}_Posterior_only.png"))

    df.to_pickle(os.path.join(output_path,f"results.pkl"))
    #print(f"Kernel: {name}")
    if isnotebook():
        display(HTML(pd.DataFrame(kernel.get_params()).to_html()))
        display(HTML(df.head().to_html()))
    else:
        #print(kernel.get_params())
        print(df)

    #fig1 = GPR.GPR.plot_n_dimensions(sample_parameters, training_x, training_y, feature_list)
    #fig1.savefig(os.path.join(output_path,'features.png'))


    #:}}}


