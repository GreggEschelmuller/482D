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

def calc_error(mu1, sigma1, mu2, sigma2):
    j1 = 1 / (sigma1**2)  # Precision for the visuial estimate
    j2 = 1 / (sigma2**2)  # Precision for the proprioceptive estimate

    mu3, sigma3 = cue_combination([mu1, mu2], [j1, j2])

    return mu3

def state_space(hand_a, hand_p, target, A, B):
    return A*hand_a + B*(target - hand_p)

def plot_gaussians(mu1, sigma1, mu2, sigma2, mu3, sigma3, hand):
    # Generate x values
    x = np.linspace(-5, 10, 1000)
    # Compute the PDF (Probability Density Function) of the Gaussians
    pdf1 = gaussian_pdf(x, mu1, sigma1)
    pdf2 = gaussian_pdf(x, mu2, sigma2)
    pdf3 = gaussian_pdf(x, mu3, sigma3)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(x, pdf1, label='Visual Estimate')
    plt.plot(x, pdf2, label='Proprioceptive Estimate')
    plt.plot(x, pdf3, label='Perceived Hand')
    plt.fill_between(x, pdf1, alpha=0.2)
    plt.fill_between(x, pdf2, alpha=0.2)
    plt.fill_between(x, pdf3, alpha=0.2, linestyle='--')
    plt.axvline(mu1, color='blue', linestyle='-', label='Target Position')
    plt.axvline(hand, color='orange', linestyle='--', label='Actual Hand')
    plt.axvline(mu3, color='green', linestyle='--', label='Perceived Hand')
    plt.ylim(0, 1)
    ax = plt.gca()  # Get current axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.title('Combination of proprioceptive and visual estimates of hand position')
    plt.xlabel('Elbow Angle')
    plt.ylabel('Probability Density')
    plt.legend(frameon=False)
    plt.savefig('cue-combination_trial2.svg')
    plt.show()


# Define the parameters for the two Gaussians
mu1, sigma1 = 0, 0.5  # Mean and standard deviation for the visual estimate
mu2, sigma2 = 4, 1.5  # Mean and standard deviation for the proprioceptive estimate
bias = mu2 - mu1
hand_actual = [mu2 - bias]

mu3, sigma3 = cue_combination([mu1, mu2], [1/sigma1**2, 1/sigma2**2])
# plot_gaussians(mu1, sigma1, mu2, sigma2, mu3, sigma3, hand_actual[0])
perceived_hand = [mu3]
v_sigmas = [sigma1]
mus = [mu2]

trials = 50
A = 0.97
B = 0.2

for i in range(trials):
    sigma1 += 0.05
    v_sigmas.append(sigma1)
    mu3, sigma3 = cue_combination([mu1, mus[-1]], [1/sigma1**2, 1/sigma2**2])
    perceived_hand.append(mu3)
    error =  B*(0 - perceived_hand[-1])
    mu2 += error
    mus.append(mu2)
    hand_actual.append(mu2 - bias)



plt.figure(figsize=(10,7))
plt.subplot(2,1,1)
plt.plot(perceived_hand, marker='.', linestyle='none', color='green', label='Perceived Hand')
plt.plot(hand_actual, marker='.', linestyle='none', color='red', label='Actual Hand')
plt.plot(mus, marker='.', linestyle='none', color='blue', label='Proprioceptive Estimate')
plt.xlabel('Trials')
plt.ylabel('Error')
plt.title(f'Error over Trials \n Learning Rate = {B}, $\sigma$ execution = {0.05}')
plt.legend(frameon=False)
ax = plt.gca()  # Get current axes
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.subplot(2,1,2)
plt.plot(v_sigmas, marker='.', linestyle='none', color='blue', label='Visual Sigma')
plt.xlabel('Trials')
plt.ylabel('$\sigma$')
plt.legend(frameon=False)
ax = plt.gca()  # Get current axes
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('state_space_sim3.svg')
plt.show()



# perceived_hand = [mu2]
# hand_actual = [mu2 - bias]
# trials = 50
# A = 0.97
# B = 0.2

# for i in range(trials):
#     hand = state_space(perceived_hand[-1], perceived_hand[-1], 0, A, B)
#     perceived_hand.append(hand)
#     hand_actual.append(perceived_hand[-1] - bias)

# print(perceived_hand)
# plt.figure(figsize=(10,7))
# plt.plot(perceived_hand, marker='.', linestyle='none', color='green', label='Perceived Hand')
# plt.plot(hand_actual, marker='.', linestyle='none', color='red', label='Actual Hand')
# plt.xlabel('Trials')
# plt.ylabel('Error')
# plt.title(f'Error over Trials \n A = {A}, B = {B}')
# plt.legend(frameon=False)
# ax = plt.gca()  # Get current axes
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# plt.savefig('state_space_sim.svg')
# plt.show()

