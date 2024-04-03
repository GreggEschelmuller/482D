import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


class CueCombinationModel:
    def __init__(self):
        """
        Initializes the cue combination model without any data.
        """
        self.has_prior = False
        self.posterior_mu = None
        self.posterior_sigma = None
        self.cues = []  # Store cues as (mean, sigma)
        self.pmes = []

    # Getter methods
    def get_pme(self):
        return self.posterior_mu

    def get_variance(self):
        return self.posterior_sigma**2

    def get_sigma(self):
        return self.posterior_sigma

    def get_precision(self):
        return 1 / (self.posterior_sigma**2)

    def get_cues(self):
        return self.cues

    def get_all_pmes(self):
        return self.pmes

    # Setter methods
    def set_pme(self, mu):
        self.posterior_mu = mu
        return None

    def set_variance(self, var):
        self.posterior_sigma = var
        return None

    def update_prior_state(self, state):
        self.has_prior = state
        return None

    def update_pmes(self, pme):
        self.pmes.append(pme)

    # Methods
    def add_cue(self, cue_mu, cue_sigma):
        """
        Adds a single cue and updates the model's belief (posterior).
        If no prior exists, the cue is treated as the initial prior.

        cue_mu: mean of the cue
        cue_sigma: standard deviation of the cue
        """
        self.cues.append((cue_mu, cue_sigma))  # Store cue

        if not self.has_prior:
            # Treat the first cue as the prior
            self.set_pme(cue_mu)
            self.set_variance(cue_sigma)
            self.update_prior_state(True)
            self.update_pmes(self.get_pme())
        else:
            # Compute the precision (inverse variance) of prior and cue
            prior_precision = 1 / (self.get_variance())
            prior_mu = self.get_pme()
            cue_precision = 1 / (cue_sigma**2)

            # Update posterior precision and mean
            posterior_precision = prior_precision + cue_precision
            self.set_pme(
                ((prior_precision * prior_mu) + (cue_precision * cue_mu))
                / (cue_precision + prior_precision)
            )
            self.set_variance(np.sqrt(1 / posterior_precision))
            self.update_pmes(self.get_pme())

    def add_cues(self, cues):
        """
        Adds multiple cues and updates the model's belief (posterior).
        Each cue is a tuple containing its mean and standard deviation.

        cues: list of tuples, where each tuple is (cue_mu, cue_sigma)
        """
        for cue_mu, cue_sigma in cues:
            self.add_cue(cue_mu, cue_sigma)

    def plot_posterior_and_cues(self):
        """
        Plots the current posterior distribution along with the cues as distributions.
        """
        if not self.has_prior:
            print("No cues have been added yet.")
            return

        fig, ax = plt.subplots(figsize=(10, 5))

        # Plot each cue as a distribution
        for cue_mu, cue_sigma in self.get_cues():
            x_cue = np.linspace(cue_mu - 3 * cue_sigma, cue_mu + 3 * cue_sigma, 1000)
            y_cue = norm.pdf(x_cue, cue_mu, cue_sigma)
            ax.plot(
                x_cue,
                y_cue,
                color="lightgray",
                label=(
                    "Cues"
                    if "Cues" not in plt.gca().get_legend_handles_labels()[1]
                    else ""
                ),
            )

        # Generate x values for the posterior
        x_posterior = np.linspace(
            self.get_pme() - (3 * self.get_variance()),
            self.get_pme() + (3 * self.get_variance()),
            1000,
        )
        # Compute the PDF of the posterior
        y_posterior = norm.pdf(x_posterior, self.get_pme(), self.get_variance())
        ax.plot(x_posterior, y_posterior, label="Posterior", color="blue")

        ax.set_title("Posterior Distribution with Cue Distributions")
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend()
        return fig, ax

    def plot_most_recent(self):

        if not self.has_prior:
            print("No cues have been added yet.")
            return
        fig, ax = plt.subplots(figsize=(10, 5))

        # Plot each cue as a distribution
        for cue_mu, cue_sigma in self.get_cues()[-2:]:
            x_cue = np.linspace(cue_mu - 3 * cue_sigma, cue_mu + 3 * cue_sigma, 1000)
            y_cue = norm.pdf(x_cue, cue_mu, cue_sigma)
            ax.plot(
                x_cue,
                y_cue,
                color="lightgray",
                label=(
                    "Cues"
                    if "Cues" not in plt.gca().get_legend_handles_labels()[1]
                    else ""
                ),
            )

        # Generate x values for the posterior
        x_posterior = np.linspace(
            self.get_pme() - (3 * self.get_variance()),
            self.get_pme() + (3 * self.get_variance()),
            1000,
        )
        # Compute the PDF of the posterior
        y_posterior = norm.pdf(x_posterior, self.get_pme(), self.get_variance())
        ax.plot(x_posterior, y_posterior, label="Posterior", color="blue")

        ax.set_title("Posterior Distribution with Cue Distributions")
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend()
        return fig, ax
