# Libraries:{{{
from .decorators import multiprocess
from .decorators import Manager
from itertools import combinations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings("ignore")
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF,Matern)
from sklearn import decomposition
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ShuffleSplit
#https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html#sklearn.model_selection.ShuffleSplit
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics
#:}}}

# k_fold_cross_val{{{
def k_fold_cross_val(estimator, X, y, folds=3):
    """scikit-learn k-fold cross-validation"""
    # Perform a cross-validation estimate of the coefficient of determination using
    # the cross_validation module using all CPUs available on the machine

    from sklearn.model_selection import cross_val_score, KFold
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
            estimator.fit(X_train, y_train)
            posterior, posterior_std = estimator.predict(X_test, return_std=True)
            results.append({"fold": fold,
                            "train index": train_index,
                            "test index": test_index,
                            "training size": len(X_test),
                            "R2": estimator.score(X_test, y_test),
                            "kernel": estimator.kernel_,
                            "hyperparameters": estimator.kernel_.hyperparameters,
                            "bounds": estimator.kernel_.bounds,
                            "alpha": estimator.alpha_,
                            "theta": estimator.kernel_.theta,
                            "LML": estimator.log_marginal_likelihood_value_,
                            "posterior": posterior,
                            "posterior_std": posterior_std,
                            "y_test": y_test,
                            "R^2": np.corrcoef(y_test, posterior)[0][1]**2,
                            "MAE": metrics.mean_absolute_error(y_test,  posterior),
                            "MSE": metrics.mean_squared_error(y_test,  posterior),
                            "RMSE": np.sqrt(metrics.mean_squared_error(y_test,  posterior)),
                            })
        return pd.DataFrame(results[:])
# }}}

# plot_PCs{{{
def plot_PCs(training_set, feature_list, n_components=3, verbose=False):
    np.random.seed(5)
    fig = plt.figure()
    fig.set_figheight(10);fig.set_figwidth(10)
    if n_components == 3:
        plt.clf()
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
        plt.cla()
    else:
       ax = plt.subplot()

    pca = decomposition.PCA(n_components)
    #training_set[
    t = training_set[feature_list]
    data_x, data_y = t.drop("pKa", axis=1).to_numpy(), t['pKa'].to_numpy()

    pca.fit(data_x)
    if verbose:
        print(f"PCA (n_features) = {pca.n_features_}")
        print(f"PCA (n_components) = {pca.n_components_}")
        print(f"PCA (n_samples) = {pca.n_samples_}")

    X = pca.transform(data_x)


    #for name, label in [('Setosa', -40), ('Versicolour', 1), ('Virginica', 2)]:
    #    ax.text3D(X[data_y == label, 0].mean(),
    #              X[data_y == label, 1].mean() + 1.5,
    #              X[data_y == label, 2].mean(), name,
    #              horizontalalignment='center',
    #              bbox=dict(alpha=.9, edgecolor='w', facecolor='w'))

    # Reorder the labels to have colors matching the cluster results
    #y = np.choose(data_y, [1, 2, 0]).astype(np.float)
    if n_components == 3:
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=data_y, cmap=plt.cm.nipy_spectral, edgecolor='k',)# s=data_y*3)
        ax.set_zlabel(r"$PC_{3}$")
    else:
        ax.scatter(X[:, 0], X[:, 1], c=data_y, cmap=plt.cm.nipy_spectral, edgecolor='k',)# s=data_y*3)
    #ax.w_xaxis.set_ticklabels([])
    ax.set_xlabel(r"$PC_{1}$")
    #ax.w_yaxis.set_ticklabels([])
    ax.set_ylabel(r"$PC_{2}$")
    #ax.w_zaxis.set_ticklabels([])
    fig.tight_layout()#pad=0.4, w_pad=0.5, h_pad=1.0)
    fig.show()
    return X,fig
# }}}

# get_correlation{{{
def get_correlation(df):
    lower_lim, upper_lim = -5, 15
    limit = np.array([lower_lim, upper_lim])
    df["R^2"] = np.corrcoef(df["Actual"], df["posterior"])[0][1]**2
    df["MAE"] = metrics.mean_absolute_error(df["Actual"],  df["posterior"])
    df["MSE"] = metrics.mean_squared_error(df["Actual"],  df["posterior"])
    df["RMSE"] = np.sqrt(metrics.mean_squared_error(df["Actual"],  df["posterior"]))
    ax = df.plot.scatter("posterior", "Actual", )
    fig = ax.get_figure()
    #fig.axes[0].plot([np.min(df["posterior"]), np.max(df["posterior"])],
    #                 [np.min(df["Actual"]), np.max(df["Actual"])], color="k")
    fig.axes[0].plot(limit, limit, color="k")
    fig.axes[0].fill_between(limit, limit+1.0, limit-1.0, facecolor='wheat', alpha=0.5)
    fig.axes[0].set_xlim(limit)
    #fig.axes[0].set_xlim(np.min(pred_y),np.max(pred_y))
    fig.axes[0].set_ylim(limit)
    #fig.axes[0].set_ylim(np.min(test_y),np.max(test_y))
    #fig.axes[0].set_xlim(np.min(df["posterior"]), np.max(df["posterior"]))
    #fig.axes[0].set_ylim(np.min(df["Actual"]), np.max(df["Actual"]))
    posx,posy = np.min(df["posterior"])+0.1, np.max(df["Actual"]) - 3
    string = r'''$N$ = %i
$R^{2}$ = %0.3f
MAE = %0.3f
MSE = %0.3f
RMSE = %0.3f'''%(len(df["Actual"]), df["R^2"][0], df["MAE"][0], df["MSE"][0], df["RMSE"][0])
    fig.axes[0].annotate(string, xy=(posx,posy), xytext=(posx,posy))
    return df, fig
# }}}

# get_regression_stats{{{
def get_regression_stats(regressor, train_x, train_y, test_x, test_y):
    #https://scikit-learn.org/0.16/modules/model_evaluation.html#regression-metrics

    lower_lim, upper_lim = -5, 15
    limit = np.array([lower_lim-10, upper_lim+10])
    df = pd.DataFrame()
    regressor.fit(train_x, train_y)
    pred_y = regressor.predict(test_x)
    df["Observed pKa"] = test_y
    df["Predicted pKa"] = pred_y
    df["R^2"] = np.corrcoef(test_y,pred_y)[0][1]**2
    df["MAE"] = metrics.mean_absolute_error(test_y, pred_y)
    df["MSE"] = metrics.mean_squared_error(test_y, pred_y)
    df["RMSE"] = np.sqrt(metrics.mean_squared_error(test_y, pred_y))
    ax = df.plot.scatter("Predicted pKa", "Observed pKa", )
    fig = ax.get_figure()
    fig.axes[0].plot(limit, limit, color="k")
    fig.axes[0].fill_between(limit, limit+1.0, limit-1.0, facecolor='wheat', alpha=0.5)
    fig.axes[0].set_xlim(limit)
    #fig.axes[0].set_xlim(np.min(pred_y),np.max(pred_y))
    fig.axes[0].set_ylim(limit)
    #fig.axes[0].set_ylim(np.min(test_y),np.max(test_y))
    posx,posy = np.min(df["Predicted pKa"])+0.1, np.max(df["Observed pKa"]) - 3
    string = r'''$N$ = %i
$R^{2}$ = %0.3f
MAE = %0.3f
MSE = %0.3f
RMSE = %0.3f'''%(len(df["Observed pKa"]), df["R^2"][0], df["MAE"][0], df["MSE"][0], df["RMSE"][0])
    fig.axes[0].annotate(string, xy=(posx,posy), xytext=(posx,posy))
    return df, fig
# }}}

# get_feature_importance{{{
def get_feature_importance(training_set, y_key='pKa', test_size=0.1, random_state=0):
    # Feature Importance with Extra Trees Classifier
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split

    df = pd.DataFrame()
    data_x, data_y = training_set.drop(y_key, axis=1), training_set[y_key]
    lab_enc = preprocessing.LabelEncoder()
    data_encoded = lab_enc.fit_transform(data_y)
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_encoded,
            test_size=test_size, random_state=random_state)

    # feature extraction
    model = ExtraTreesClassifier(n_estimators=100)
    model.fit(train_x, train_y)
    df["Importance"] = model.feature_importances_
    df["Feature"] = list(training_set.keys())[:-1]
    df = df.sort_values('Importance', ascending=True)
    return df
# }}}






