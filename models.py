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

    def add_cue(self, cue_mu, cue_sigma):
        """
        Adds a single cue and updates the model's belief (posterior).
        If no prior exists, the cue is treated as the initial prior.
        :param cue_mu: mean of the cue (likelihood)
        :param cue_sigma: standard deviation of the cue (likelihood)
        """
        self.cues.append((cue_mu, cue_sigma))  # Store cue

        if not self.has_prior:
            # Treat the first cue as the prior
            self.posterior_mu = cue_mu
            self.posterior_sigma = cue_sigma
            self.has_prior = True
        else:
            # Compute the precision (inverse variance) of prior and cue
            prior_precision = 1 / (self.posterior_sigma**2)
            cue_precision = 1 / (cue_sigma**2)

            # Update posterior precision and mean
            posterior_precision = prior_precision + cue_precision
            self.posterior_mu = (
                self.posterior_mu * cue_precision + cue_mu * prior_precision
            ) / posterior_precision
            self.posterior_sigma = np.sqrt(1 / posterior_precision)

    def add_cues(self, cues):
        """
        Adds multiple cues and updates the model's belief (posterior).
        Each cue is a tuple containing its mean and standard deviation.
        :param cues: list of tuples, where each tuple is (cue_mu, cue_sigma)
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
        for cue_mu, cue_sigma in self.cues:
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
            self.posterior_mu - 3 * self.posterior_sigma,
            self.posterior_mu + 3 * self.posterior_sigma,
            1000,
        )
        # Compute the PDF of the posterior
        y_posterior = norm.pdf(x_posterior, self.posterior_mu, self.posterior_sigma)
        ax.plot(x_posterior, y_posterior, label="Posterior", color="blue")

        ax.set_title("Posterior Distribution with Cue Distributions")
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend()
        return fig, ax

    def get_pme(self):
        return self.posterior_mu

    def get_variance(self):
        return self.posterior_sigma**2

    def get_precision(self):
        return 1 / (self.posterior_sigma**2)


def main():
    # Code to execute when the script is run directly
    print("Hello, world!")


if __name__ == "__main__":
    main()
