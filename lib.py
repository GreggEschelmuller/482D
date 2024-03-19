import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def gaussian_pdf(y, mu, sigma):
    const = 1 / (np.sqrt(2 * np.pi * (sigma**2)))
    exp = np.exp(-(((y - mu) ** 2) / (2 * (sigma**2))))
    return const * exp


def calc_likelihood(mu, sigma, s):
    return gaussian_pdf(s, mu, sigma)


def plot_gaussian(data, s, label="Gaussian"):
    fig, ax = plt.subplots()
    ax.plot(s, data, color="forestgreen", label=label)
    ax.set_xlabel("stimulus values")
    ax.set_ylabel("probability density")
    ax.set_title("Prior")
    ax.legend(frameon=False)
    sns.despine()
    return fig, ax


def calc_posterior(prior, likelihood, step_size):
    protoposterior = prior * likelihood
    posterior = protoposterior / np.sum(protoposterior)
    posterior /= step_size
    return posterior


def plot_posterior(prior, likelihood, posterior, s):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(s, prior, color="forestgreen", label="Prior $p(s)$")
    ax.plot(s, likelihood, color="darkblue", label="Likelihood $p(x|s)$")
    ax.plot(s, posterior, color="teal", label="Posterior $p(s|x)$")
    ax.set_xlabel("stimulus values")
    ax.set_ylabel("probability density")
    ax.set_title("Prior")
    ax.legend(frameon=False)
    sns.despine()
    return fig, ax


def cue_comb(cue1, cue2, step_size):
    protoposterior = cue1 * cue2
    posterior = protoposterior / np.sum(protoposterior)
    posterior /= step_size
    return posterior


def cue_combination(cues, step_size):
    """
    cues = dictionary with the keys as the cue number (1, 2, 3, 4), and values as the pdf
    step_size = step size for stimulus generation
    """
    cues[0]
