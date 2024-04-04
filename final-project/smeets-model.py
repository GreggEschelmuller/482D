import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def plot_gaussians_subplot(
    bias_v, sigma_v, bias_p, sigma_p, perceived_positions, perceived_sigmas, plot_trials
):
    """
    Plots the Gaussian distributions for visual and proprioceptive estimates,
    and their combined posterior for specified trials using subplots.
    """
    x = np.linspace(-10, 10, 1000)
    num_plots = len(plot_trials)

    fig, axs = plt.subplots(num_plots, 1, figsize=(8, 8))

    for i, trial in enumerate(plot_trials):
        # Calculate the perceived position Gaussian as a simple demonstration

        perceived_gaussian = norm.pdf(
            x, perceived_positions[trial], perceived_sigmas[trial]
        )

        vis_gaussian = norm.pdf(x, bias_v, sigma_v[trial])
        prop_gaussian = norm.pdf(x, bias_p, sigma_p[trial])

        axs[i].plot(x, vis_gaussian, label="Visual Estimate", color="blue")
        axs[i].plot(x, prop_gaussian, label="Proprioceptive Estimate", color="green")
        axs[i].plot(
            x,
            perceived_gaussian,
            label="Perceived Position",
            color="red",
            linestyle="--",
        )
        axs[i].set_title(f"Trial {trial+1}")
    plt.tight_layout()
    plt.legend()
    plt.show()


# Parameters for simulation
bias_v, bias_p = 2, -4
sigma_v, sigma_p = 0.5, 1.5
decay_v = 0.5
decay_p = 0.1
movements = 50
plot_trials = [0, 5, 25, 49]  # Trials at which to plot the Gaussians and posterior

j_v = 1 / sigma_v**2
j_p = 1 / sigma_p**2
perceived_position = (j_v * bias_v + j_p * bias_p) / (j_v + j_p)
perceived_sigma = np.sqrt(1 / (j_v + j_p))

perceived_sigmas = [perceived_sigma]
drift_trajectory = [perceived_position]
sigma_v_values = [sigma_v]  # To keep track of visual variance over time
sigma_p_values = [sigma_p]

for movement in range(movements):
    sigma_v = np.sqrt(sigma_v**2 + decay_v**2)
    sigma_p = np.sqrt(sigma_p**2 + decay_p**2)
    sigma_v_values.append(sigma_v)  # Update visual variance tracking
    sigma_p_values.append(sigma_p)  # update proprioceptive variance tracking

    j_v = 1 / sigma_v**2
    j_p = 1 / sigma_p**2
    perceived_position = (j_v * bias_v + j_p * bias_p) / (j_v + j_p)
    perceived_sigma = np.sqrt(1 / (j_v + j_p))
    drift_trajectory.append(perceived_position)
    perceived_sigmas.append(perceived_sigma)

# Plot Gaussians and posterior for specified trials using subplots
plot_gaussians_subplot(
    bias_v,
    sigma_v_values,
    bias_p,
    sigma_p_values,
    drift_trajectory,
    perceived_sigmas,
    plot_trials,
)

# Plotting
plt.figure(figsize=(10, 6))
plt.axhline(bias_v, linestyle="--")
plt.axhline(bias_p, linestyle="--")
plt.plot(drift_trajectory, marker="o", linestyle="-", color="red")
plt.title("Simulated Drift in Perceived Hand Position with Biases")
plt.xlabel("Movement Number")
plt.ylabel("Perceived Position (deg) relative to the target")
plt.grid(True)
plt.show()
