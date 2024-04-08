import numpy as np
import matplotlib.pyplot as plt
import numpy as np

def cue_combination(mu, prec):
    posterior_mu = ((mu[0] * prec[0]) + (mu[1] * prec[1])) / (prec[0] + prec[1])
    sigma = np.sqrt(1/(prec[0] + prec[1]))

    return posterior_mu, sigma

def gaussian_pdf(y, mu, sigma):
    const = 1 / (np.sqrt(2 * np.pi * (sigma**2)))
    exp = np.exp(-(((y - mu) ** 2) / (2 * (sigma**2))))
    return const * exp

def plot_gaussians(mu1, sigma1, mu2, sigma2, mu3, sigma3):
    # Generate x values
    x = np.linspace(-5, 10, 1000)
    # Compute the PDF (Probability Density Function) of the Gaussians
    pdf1 = gaussian_pdf(x, mu1, sigma1)
    pdf2 = gaussian_pdf(x, mu2, sigma2)
    pdf3 = gaussian_pdf(x, mu3, sigma3)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(x, pdf1, label='Proprioceptive Estimate')
    plt.plot(x, pdf2, label='Visual Estimate')
    plt.plot(x, pdf3, label='Comvined Estimate')
    plt.fill_between(x, pdf1, alpha=0.2)
    plt.fill_between(x, pdf2, alpha=0.2)
    plt.fill_between(x, pdf3, alpha=0.2, linestyle='--')
    # plt.axvline(mu1, color='blue', linestyle='--', label='Prop Estimate')
    # plt.axvline(mu2, color='orange', linestyle='--', label='Visual Estimate')
    # plt.axvline(mu3, color='green', linestyle='--', label='Perceived Hand')
    plt.ylim(0, 1)
    ax = plt.gca()  # Get current axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.title('')
    plt.xlabel('Elbow Angle')
    plt.ylabel('Probability Density')
    plt.legend(frameon=False)
    plt.savefig('smeets_hand_baseline.svg')
    plt.show()

def calc_hand(prop_mu, vis_mu, prop_sig, vis_sig, ex_sig, n):
    factor1 = (vis_sig+n*ex_sig)/(prop_sig+vis_sig+n*ex_sig)
    factor2 = (prop_sig)/(prop_sig+vis_sig+n*ex_sig)
    return factor1*prop_mu + factor2*vis_mu

def calc_target(prop_mu, vis_mu, prop_sig, vis_sig, ex_sig, n):
    factor1 = (vis_sig)/(prop_sig+vis_sig+n*ex_sig)
    factor2 = (prop_sig+n*ex_sig)/(prop_sig+vis_sig+n*ex_sig)
    return factor1*prop_mu + factor2*vis_mu

def calc_bias(p_mu, v_mu, p_sig, v_sig, ex_sig, n):
    hand = calc_hand(p_mu, v_mu, p_sig, v_sig, ex_sig, n)
    target = calc_target(p_mu, v_mu, p_sig, v_sig, ex_sig, n)
    return hand - target


prop_sig = 1.5
vis_sig = 1.5
ex_sig = 1

prop_mu = 5
vis_mu = 0
bias = prop_mu - vis_mu

biases = []

for i in range(100):
    bias_est = calc_bias(prop_mu, vis_mu, prop_sig, vis_sig, ex_sig, i)
    biases.append(bias_est)

plt.figure(figsize=(10, 6))
plt.plot(biases, marker='.', linestyle='none', color='green', label='Perceived Hand')
plt.legend(frameon=False)
plt.axhline(bias, color='blue', linestyle='--', label='Actual Bias')
plt.title('Bias over time')
plt.xlabel('Trials')
plt.ylabel('Bias')
ax = plt.gca()  # Get current axes
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig('smeets_bias.svg')
plt.show()

# perceived_hand, sig = cue_combination([prop_mu_h, vis_mu_h],[1/prop_sig**2, 1/vis_sig**2])
# plot_gaussians(prop_mu_h, prop_sig, vis_mu_h, vis_sig, perceived_hand, sig)