import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def plot_gaussians_subplot(
    bias_v, sigma_v, bias_p, sigma_p, perceived_positions, plot_trials
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
        
        sigma_perceived = (sigma_v[trial] + sigma_p) / 2
        perceived_gaussian = norm.pdf(x, perceived_positions[trial], sigma_perceived)

        vis_gaussian = norm.pdf(x, bias_v, sigma_v[trial])
        prop_gaussian = norm.pdf(x, bias_p, sigma_p)

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


def simulate_drift_with_subplots(
    bias_v, bias_p, sigma_v, sigma_p, sigma_ex, movements, plot_trials
):
    weight_v = 1 / sigma_v**2 / (1 / sigma_v**2 + 1 / sigma_p**2)
    weight_p = 1 / sigma_p**2 / (1 / sigma_v**2 + 1 / sigma_p**2)
    perceived_position = weight_v * bias_v + weight_p * bias_p

    drift_trajectory = [perceived_position]
    sigma_v_values = [sigma_v]  # To keep track of visual variance over time

    for movement in range(movements):
        sigma_v = np.sqrt(sigma_v**2 + sigma_ex**2)
        sigma_v_values.append(sigma_v)  # Update visual variance tracking

        weight_v = 1 / sigma_v**2 / (1 / sigma_v**2 + 1 / sigma_p**2)
        weight_p = 1 / sigma_p**2 / (1 / sigma_v**2 + 1 / sigma_p**2)
        perceived_position = weight_v * bias_v + weight_p * bias_p
        drift_trajectory.append(perceived_position)

    # Plot Gaussians and posterior for specified trials using subplots
    plot_gaussians_subplot(
        bias_v, sigma_v_values, bias_p, sigma_p, drift_trajectory, plot_trials
    )

    return drift_trajectory


# Parameters for simulation
bias_v, bias_p = 2, -4
sigma_v, sigma_p = 0.5, 1.5
sigma_ex = 0.5
movements = 50
plot_trials = [0, 5, 25, 49]  # Trials at which to plot the Gaussians and posterior

# Simulate and plot using subplots
drift_trajectory_biases = simulate_drift_with_subplots(
    bias_v, bias_p, sigma_v, sigma_p, sigma_ex, movements, plot_trials
)

# Plotting
plt.figure(figsize=(10, 6))
plt.axhline(bias_v, linestyle="--")
plt.axhline(bias_p, linestyle="--")
plt.plot(drift_trajectory_biases, marker="o", linestyle="-", color="red")
plt.title("Simulated Drift in Perceived Hand Position with Biases")
plt.xlabel("Movement Number")
plt.ylabel("Perceived Position (deg) relative to the target")
plt.grid(True)
plt.show()
