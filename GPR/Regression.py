# Import Libraries:{{{
import time, os
import numpy as np
import pandas as pd
import itertools
import scipy
from scipy.stats import norm
from scipy.optimize import minimize
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF,Matern)
from sklearn.model_selection import train_test_split
from . import statistics
import warnings
warnings.filterwarnings("ignore")
pd.options.display.max_columns = 50
pd.options.display.max_rows = 10
#:}}}

# get_prior_and_posterior: {{{
def get_prior_and_posterior(df, Regressor, sample_parameters,
        training_x, training_y, feature_list,
        random_state=0, title=False, output_path="./"):
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
    para = sample_parameters

    # Get Figure
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_figheight(4);fig.set_figwidth(18)
    # Plot prior

    #X = np.array([[training[np.random.randint(len(training))]] for i in range(len(x))])
    y = para["function"]()

    #y = np.array([(y[:,i]-y[:,i].mean())/y[:,i].var() for i in range(len(y[0]))])
    #y = y.transpose()

    df["prior"], df["prior_std"] = Regressor.predict(y, return_std=True)
    prior, prior_cov = Regressor.predict(y, return_cov=True)
    #print(Regressor.kernel.n_dims)
    #print(f'df["prior"] = {df["prior"]}')
    ax1.plot(x, df["prior"], 'k', lw=3, zorder=9)
    ax1.fill_between(x, df["prior"] - df["prior_std"], df["prior"] + df["prior_std"],
                     alpha=0.2, color='k')

    y_samples = Regressor.sample_y(y, para["nsamples"], random_state)

    plot_kernel(X=x, y=y_samples, Σ=prior_cov, description="kernel",
            xlim=(min(x), max(x)), scatter=False, rotate_x_labels=False,
            figname="kernel_plot_prior.pdf", output_path=output_path)

    ax1.plot(x, y_samples, lw=1)
    ax1.set_xlim(min(x), max(x))
    ax1.set_ylim(-10,10)#np.min(y), np.max(y))
    ax1.set_title("Prior (kernel:  %s)" % Regressor.kernel, fontsize=12)
    ############################################################################
    # Generate df and fit GP
    rng = np.random.RandomState(4)
    #print(f'y.shape = {y.shape}')
    #print(f'y = {y}')
    stime = time.time()
    # Fit to data using Maximum Likelihood Estimation of the parameters
    Regressor.fit(training_x, training_y) # fit( training_set_X, training_set_y )
    print(f"Regressor.score(training_x, training_y) = {Regressor.score(training_x, training_y)}")
    print(f"Time for {str(Regressor.__str__).split()[3]} fitting: %.3f s" % (time.time() - stime))
    ############################################################################
    # Plot posterior
    df["posterior"], df["posterior_std"] = Regressor.predict(y, return_std=True)
    posterior, posterior_cov = Regressor.predict(y,  return_cov=True)

    # Plot posterior for each dimension :{{{
    fig1, axx = plt.subplots(len(y[0]), 1) # Get Figure
    fig1.set_figheight(16);fig1.set_figwidth(12)

    #NOTE: standardization?
    #training_x = np.array([(training_x[:,i]-training_x[:,i].mean())/training_x[:,i].std() for i in range(len(training_x[0]))])
    #training_x = training_x.transpose()
    for i in range(len(y[0])):
        ax_x = axx[i]
        #y_samples = Regressor.sample_y(y, para["nsamples"])
        x_,y_ = training_x[:,i], training_y #FIXME
        ax_x.scatter(x_, y_, c='r', s=50, zorder=10, edgecolors=(0, 0, 0), label=f'{feature_list[i]}')
        ax_x.set_xlim(np.min(x_), np.max(x_))
        ax_x.set_ylim(np.min(y_), np.max(y_))
        ax_x.legend(framealpha=0.0, markerscale=0)
    fig1.tight_layout(pad=1., w_pad=0.5, h_pad=0.5)
    fig1.savefig(os.path.join(output_path,f"Posterior_all_{len(y[0])}_dimensions.png"))
    #:}}}

    #print(f'df["posterior"] = {df["posterior"]}')
    ax2.plot(x, df["posterior"], 'k', lw=3, zorder=9)
    ax2.fill_between(x, df["posterior"] - df["posterior_std"],
            df["posterior"] + df["posterior_std"],
                     alpha=0.2, color='k')
    y_samples = Regressor.sample_y(y, para["nsamples"], random_state)

    ndim = len(training_x[0])
    # Compute the mean of the sample.
    y_mean = np.apply_over_axes(func=np.mean, a=y_samples, axes=1).squeeze()
    # Compute the standard deviation of the sample.
    y_std = np.apply_over_axes(func=np.std, a=y_samples, axes=1).squeeze()
    ax2.plot(x, y_mean, '--k', lw=3, zorder=9)
    ax2.fill_between(x, y_mean-2*y_std, y_mean+2*y_std, alpha=0.2, color='b')


    plot_kernel(X=x, y=y_samples, Σ=posterior_cov, description="kernel",
            xlim=(min(x), max(x)), scatter=False, rotate_x_labels=False,
            figname="kernel_plot_posterior.pdf", output_path=output_path)

    # Obtain optimized kernel parameters
    l = Regressor.kernel_.k2.get_params()['length_scale']
    sigma_f = np.sqrt(Regressor.kernel_.k1.get_params()['constant_value'])

    #print(f'y_samples.shape = {y_samples.shape}')
    #print(f'y_samples = {y_samples}')
    #ax2.plot(x, y_samples, lw=1)
#    ax2.scatter(x, df["Actual"], marker='*', color="c")#, lw=2, zorder=9)
    #ax2.scatter(x[:,np.newaxis], y, c='r', s=50, zorder=10, edgecolors=(0, 0, 0))
    #x_,y_ = training_x[:,1], training_y #FIXME
    #print(x_.shape)
    #print(y_.shape)

    #get_log_marginal_likelihood(gp=Regressor, thetas=None, figname="log_marginal_likelihood.pdf")


    #ax2.scatter(x_,y_, c='r', s=50, zorder=10, edgecolors=(0, 0, 0))
    #ax2.scatter(y, y_samples, c='r', s=50, zorder=10, edgecolors=(0, 0, 0))
    ax2.set_xlim(min(x), max(x))
    #ax2.set_ylim(-3, 3)
    ax2.set_ylim(np.min(training_y), np.max(training_y))
    ax2.set_title(r"""Posterior (kernel: %s)
    Log-Likelihood: %.3f"""%(Regressor.kernel_,
        Regressor.log_marginal_likelihood(Regressor.kernel_.theta)),fontsize=12)
    fig.tight_layout()#pad=0.4, w_pad=0.5, h_pad=1.0)
    if title:
        fig.title(title)

    df = pd.DataFrame(df)
    fig.show()
    return df, fig
# }}}

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
    para = sample_parameters
    # Get Figure
    fig, (ax2) = plt.subplots(1)
    fig.set_figheight(4);fig.set_figwidth(10)
    y = para["function"]()



    ############################################################################
    # NOTE: Prior
    ############################################################################
    df["prior"], df["prior_std"] = Regressor.predict(y, return_std=True)
    prior, prior_cov = Regressor.predict(y, return_cov=True)
    y_samples = Regressor.sample_y(y, para["nsamples"], random_state)
    plot_kernel(X=x, y=y_samples, Σ=prior_cov, description="kernel",
            xlim=(min(x), max(x)), scatter=False, rotate_x_labels=False,
            figname="kernel_plot_prior.pdf", output_path=output_path)
    ############################################################################

    # Generate df and fit GP
    rng = np.random.RandomState(4)
    itime = time.time()

    rounds_of_opt = 10 #FIXME
    #for n in range(rounds_of_opt):

    # Fit to data using Maximum Likelihood Estimation of the parameters
    Regressor.fit(training_x, training_y) # fit( training_set_X, training_set_y )
    LML = r"""Posterior (kernel: %s)
Log-Likelihood: %.3f"""%(Regressor.kernel_,
        Regressor.log_marginal_likelihood(Regressor.kernel_.theta))
    print(LML)
    print("\n")
    #print(Regressor.kernel_.hyperparameters)
    hyperpara = Regressor.kernel_.__dict__
    #print(Regressor.kernel_.get_params)
    #exit()
    print(f"hyperparameters = {hyperpara}")
    k1 = hyperpara["k1"]
    k2 = hyperpara["k2"]
    #help(Regressor.kernel_)
    #print(k2.length_scale)
    #print(type(k2))
    #print(Regressor.kernel_.theta)
    #Regressor.kernel_.set_params(dict(length_scale=k2.length_scale))
    #Regressor.kernel_.set_params(k2.length_scale)
    #hyperpara = Regressor.kernel_.__dict__
    #print(f"corrected hyperparameters = {hyperpara}")
    #Regressor.fit(training_x, training_y) # fit( training_set_X, training_set_y )
    #hyperpara = Regressor.kernel_.__dict__
    #print(f"final hyperparameters = {hyperpara}")
    #exit()
    #ax2.set_title(LML,fontsize=12)


    ftime = time.time()
    print(f"{ftime-itime} for {rounds_of_opt} rounds of optimizing kernel lengths.")
    #exit()

    # Plot posterior
    df["posterior"], df["posterior_std"] = Regressor.predict(y,  return_std=True)
    posterior, posterior_cov = Regressor.predict(y,  return_cov=True)
 # NOTE: this indented comments are for test
 #   ax2.plot(x, df["posterior"], 'k', lw=3, zorder=9)
    #ax2.fill_between(x, df["posterior"] - df["posterior_std"],
    #        df["posterior"] + df["posterior_std"],
    #                 alpha=0.2, color='k')
    y_samples = Regressor.sample_y(y, para["nsamples"], random_state)


    plot_kernel(X=x, y=y_samples, Σ=posterior_cov, description="kernel",
            xlim=(min(x), max(x)), scatter=False, rotate_x_labels=False,
            figname="kernel_plot_posterior.pdf", output_path=output_path)

 # NOTE: this indented comments are for test
    data = pd.concat([pd.DataFrame(y), pd.DataFrame(df["posterior"])], axis=1)
    data = data.reindex()
    data.columns = feature_list
    corner_plot(data, figname=os.path.join(output_path,"SM07_coner_plot.png"))

    training_data = pd.concat([pd.DataFrame(training_x), pd.DataFrame(training_y)], axis=1)
    training_data = training_data.reindex()
    training_data.columns = feature_list
    corner_plot(training_data, figname=os.path.join(output_path,"training_coner_plot.png"))

    #exit()
 #   ax2.scatter(y, df["posterior"])
    ax2.fill_between(x, df["posterior"] - df["posterior_std"],
            df["posterior"] + df["posterior_std"], alpha=0.3, color='k')
    #ax2.plot(x, y_samples, lw=1)
    ax2.plot(x, df["posterior"], "k", lw=1)
    #ax2.scatter(x, df["Actual"], marker='*', color="c")#, lw=2, zorder=9)
 #   x_,y_ = training_x[:,1], training_y #FIXME
 #   ax2.scatter(x_,y_, c='r', s=50, zorder=10, edgecolors=(0, 0, 0))
    #ax2.set_ylim(np.min(training_y), np.max(training_y))
    ax2.set_xlim(np.min(x), np.max(x))
    fig.tight_layout()#pad=0.4, w_pad=0.5, h_pad=1.0)
    if title:
        fig.title(title)
    df = pd.DataFrame(df)
    fig.show()
    return df, fig

#:}}}

# plot_posterior_CO2:{{{
def plot_posterior_CO2(df, Regressor, sample_parameters, training_x, training_y,
        random_state=0, title=False):
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
    para = sample_parameters
    # Get Figure
    fig, (ax2) = plt.subplots(1)
    fig.set_figheight(4);fig.set_figwidth(18)
    y = para["function"]()
    # Generate df and fit GP
    rng = np.random.RandomState(4)
    stime = time.time()
    # Fit to data using Maximum Likelihood Estimation of the parameters
    Regressor.fit(training_x, training_y) # fit( training_set_X, training_set_y )
    ax2.set_title(r"""Posterior (kernel: %s)
    Log-Likelihood: %.3f"""%(Regressor.kernel_,
        Regressor.log_marginal_likelihood(Regressor.kernel_.theta)),fontsize=12)

    # Plot posterior
    df["posterior"], df["posterior_std"] = Regressor.predict(y,  return_std=True)
 # NOTE: this indented comments are for test
 #   ax2.plot(x, df["posterior"], 'k', lw=3, zorder=9)
    #ax2.fill_between(x, df["posterior"] - df["posterior_std"],
    #        df["posterior"] + df["posterior_std"],
    #                 alpha=0.2, color='k')
    y_samples = Regressor.sample_y(y, para["nsamples"], random_state)

 # NOTE: this indented comments are for test
    ax2.set_xlabel("Year")
    ax2.set_ylabel(r"CO$_2$ in ppm")
    ax2.plot(y, df["posterior"])
    ax2.fill_between(y[:, 0], df["posterior"] - df["posterior_std"],
            df["posterior"] + df["posterior_std"], alpha=0.3, color='k')
 #   ax2.plot(x, y_samples, lw=1)
#    ax2.scatter(x, df["Actual"], marker='*', color="c")#, lw=2, zorder=9)
 #   x_,y_ = training_x[:,1], training_y #FIXME
 #   ax2.scatter(x_,y_, c='r', s=50, zorder=10, edgecolors=(0, 0, 0))
    ax2.scatter(training_x, training_y, c="k")
    ax2.set_xlim(min(y), max(y))



    #ax2.set_ylim(np.min(training_y), np.max(training_y))
    fig.tight_layout()#pad=0.4, w_pad=0.5, h_pad=1.0)
    if title:
        fig.title(title)
    df = pd.DataFrame(df)
    fig.show()
    return df, fig

#:}}}

# plot_posterior_:{{{
def plot_posterior_(df, Regressor, sample_parameters, training_x, training_y,
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
    para = sample_parameters
    # Get Figure
    fig, (ax2) = plt.subplots(1)
    fig.set_figheight(4);fig.set_figwidth(18)
    y = para["function"]()
    # Generate df and fit GP
    rng = np.random.RandomState(4)
    stime = time.time()
    Regressor = Regressor.fit(training_x, training_y)
    posterior, posterior_cov = Regressor.predict(y, return_cov=True)
    posterior_std = np.sqrt(Regressor.kernel_.diag(y))
    #print(Regressor.log_marginal_likelihood(Regressor.kernel_.theta))

    y = sample_parameters["function"]()

    y = np.array([(y[:,i]-y[:,i].mean())/y[:,i].std() for i in range(len(y[0]))])

    #Y = np.array([(y[:,i]-y[:,i].mean())/y[:,i].std() for i in range(len(y[0]))])
    #print(Y)
    #new_y = np.matrix(posterior_cov)*np.matrix(Y.transpose())
    #print(new_y)
    #print(new_y.shape)

    #M = np.matrix(np.random.multivariate_normal(mean=posterior, cov=posterior_cov, size=para["nsamples"]))
    #print(M)
    #print(M.shape)
    #F = M*new_y
    #print(F)
    #print(F.shape)
    #print(np.mean(F, axis=0))
    #exit()

    fig1, axx = plt.subplots(len(y[0]), 2) # Get Figure
    fig1.set_figheight(16);fig1.set_figwidth(12)
    for i in range(len(y[0])):
        ax_x,ax_xx = axx[i,0],axx[i,1]
        # Sample from the posterior distribution.
        z = np.random.multivariate_normal(mean=posterior, cov=posterior_cov, size=para["nsamples"])
        z = z.T
        print(z)
        print(z.shape)
        # NOTE: right now i have (# number of transitions x # samples)
        # NOTE: TODO:  distributions of samples for each of the 10 feature spaces

        exit()

        ax_x.scatter(x_, y_, c='r', s=50, zorder=10, edgecolors=(0, 0, 0), label=f'{feature_list[i]}')


        ax_xx.hist(y_, edgecolor='k', orientation='horizontal',)
        ax_x.set_xlim(np.min(x_), np.max(x_))
        ax_x.set_ylim(np.min(y_), np.max(y_))
        ax_x.legend(framealpha=0.0, markerscale=0)
    fig1.tight_layout(pad=1., w_pad=0.5, h_pad=0.5)
    fig1.savefig(os.path.join(output_path,f"Posterior_all_{len(y[0])}_dimensions.png"))
    return fig1

#:}}}

# corner_plot:{{{
def corner_plot(data, scatter_colors="k", figname="corner_plot.png", output_path="./"):
    from pandas.plotting import scatter_matrix
    import matplotlib.pylab as pylab
    fig1, (ax) = plt.subplots(1)
    ax.xaxis.label.set_rotation(0)
    ax.yaxis.label.set_rotation(0)
    axs = scatter_matrix(data, alpha=0.2, figsize=(16, 16), diagonal='hist', c=scatter_colors,
            hist_kwds=dict(facecolor="b", edgecolor="k", linewidth=1.25, bins=15), ax=ax)
    n = len(data.columns)
    for ax_x in range(n):
        for ax_y in range(n):
            # to get the axis of subplots
            ax = axs[ax_x, ax_y]
            # to make x axis name vertical
            ax.xaxis.label.set_rotation(45)
            ax.xaxis.label.set_size(8)
            # to make y axis name horizontal
            ax.yaxis.label.set_rotation(0)
            ax.yaxis.label.set_size(8)
            # to make sure y axis names are outside the plot area
            ax.yaxis.labelpad = 20
            if ax_x == ax_y:
                mean = data[list(data.keys())[ax_y]].mean()
                print(f"{list(data.keys())[ax_y]} mean = {mean}")
                ax.axvline(x=mean, linewidth=1.5, color='r')

    #fig1 = ax.get_figure()
    fig1.tight_layout(pad=0.1, w_pad=-2, h_pad=-0.5)
    fig1.savefig(os.path.join(output_path,figname))
    return fig1
#:}}}

# plot_n_dimensions:{{{
def plot_n_dimensions(sample_parameters, training_x, training_y, feature_list, output_path="./"):

    y = sample_parameters["function"]()
    fig1, axx = plt.subplots(len(y[0]), 1) # Get Figure
    fig1.set_figheight(16);fig1.set_figwidth(12)
    for i in range(len(y[0])):
        ax_x = axx[i]
        #y_samples = Regressor.sample_y(y, para["nsamples"])
        x_,y_ = training_x[:,i], training_y #FIXME
        ax_x.scatter(x_, y_, c='r', s=50, zorder=10, edgecolors=(0, 0, 0), label=f'{feature_list[i]}')
        ax_x.set_xlim(np.min(x_), np.max(x_))
        ax_x.set_ylim(np.min(y_), np.max(y_))
        ax_x.legend(framealpha=0.0, markerscale=0)
    fig1.tight_layout(pad=1., w_pad=0.5, h_pad=0.5)
    fig1.savefig(os.path.join(output_path,f"Posterior_all_{len(y[0])}_dimensions.png"))
    return fig1
#:}}}

# plot_n_dimensions_:{{{
def plot_n_dimensions_(sample_parameters, training_x, training_y, training_set,
        feature_list, output_path="./"):

    y = sample_parameters["function"]()
    fig1, axx = plt.subplots(len(y[0]), 2) # Get Figure
    fig1.set_figheight(16);fig1.set_figwidth(12)
    for i in range(len(y[0])):
        ax_x,ax_xx = axx[i,0],axx[i,1]
        #y_samples = Regressor.sample_y(y, para["nsamples"])
        x_,y_ = training_x[:,i], training_y #FIXME
        ax_x.scatter(x_, y_, c='r', s=50, zorder=10, edgecolors=(0, 0, 0), label=f'{feature_list[i]}')

        #####################################################################
        feat = "num Ionizable groups"
        value = training_set[feat]
        cond3 = np.where(value == training_set[feat])

        subset = list(set(cond1[0]) & set(cond2[0]) & set(cond3[0]))
        train = training.iloc[subset]


        ax_xx.hist(y_, edgecolor='k', orientation='horizontal',)
        ax_x.set_xlim(np.min(x_), np.max(x_))
        ax_x.set_ylim(np.min(y_), np.max(y_))
        ax_x.legend(framealpha=0.0, markerscale=0)
    fig1.tight_layout(pad=1., w_pad=0.5, h_pad=0.5)
    fig1.savefig(os.path.join(output_path,f"Posterior_all_{len(y[0])}_dimensions.png"))
    return fig1
#:}}}

# plot_kernel:{{{
def plot_kernel(X, y, Σ, description="kernel", xlim=(-10,10), scatter=False, rotate_x_labels=False,
        figname="kernel_plot.pdf", output_path="./"):
    """Plot kernel matrix and samples."""

    fig = plt.figure(figsize=(7, 2.7))
    gs = gridspec.GridSpec(1, 1, figure=fig, wspace=0.0, hspace=0.0)

    grid_spec = gridspec.GridSpecFromSubplotSpec(
        1, 2, width_ratios=[2,1], height_ratios=[1],
        wspace=0.18, hspace=0.0, subplot_spec=gs[0])

    ax1 = fig.add_subplot(grid_spec[0])
    ax2 = fig.add_subplot(grid_spec[1])
    # Plot samples
    if scatter:
        for i in range(y.shape[1]):
            ax1.scatter(X, y[:,i], alpha=0.8, s=3)
    else:
        for i in range(y.shape[1]):
            ax1.plot(X, y[:,i], alpha=0.8)
    ax1.set_ylabel('$y$', fontsize=13, labelpad=0)
    ax1.set_xlabel('$x$', fontsize=13, labelpad=0)
    ax1.set_xlim(xlim)
    if rotate_x_labels:
        for l in ax1.get_xticklabels():
            l.set_rotation(30)
    ax1.set_title(f'Samples from {description}')
    # Plot covariance matrix
    im = ax2.imshow(Σ, cmap=cm.YlGnBu)
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.02)
    cbar = plt.colorbar(im, ax=ax2, cax=cax)
    cbar.ax.set_ylabel('$K(X,X)$', fontsize=8)
    ax2.set_title(f'Covariance matrix\n{description}')
    ax2.set_xlabel('X', fontsize=10, labelpad=0)
    ax2.set_ylabel('X', fontsize=10, labelpad=0)
    if rotate_x_labels:
        for l in ax2.get_xticklabels():
            l.set_rotation(30)
    ax2.grid(False)
    fig.tight_layout()#pad=0.4, w_pad=0.5, h_pad=1.0)
    fig.savefig(os.path.join(output_path,figname))
#:}}}

# get_stats{{{
def get_stats(regressor, training_x, training_y, test_size=0.5, random_state=0):
    train_x, test_x, train_y, test_y = train_test_split(training_x, training_y, test_size, random_state)
    dfs = statistics.get_regression_stats(regressor, train_x, train_y, test_x, test_y)
    return dfs
# }}}



