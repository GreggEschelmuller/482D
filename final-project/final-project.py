from models import CueCombinationModel
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def cue_combination(mu, prec):
    posterior_mu = ((mu[0] * prec[0]) + (mu[1] * prec[1])) / (prec[0] + prec[1])
    posterior_prec = prec[0] + prec[1]

    return posterior_mu, posterior_prec


# Define starting parameters
vis_mean = 1
vis_sig = 0.5

prop_mean = -5
prop_sig = 3

posterior_mus = []
posterior_sigs = []
vis_sigs = []
prop_sigs = []


# Trial one
for trial in range(101):
    vis_sig += 0.2
    prop_sig += 0.05
    vis_sigs.append(vis_sig)
    prop_sigs.append(prop_sig)

    pm, ps = cue_combination([vis_mean, prop_mean], [1 / vis_sig**2, 1 / prop_sig**2])
    posterior_mus.append(pm)
    posterior_sigs.append(ps)


fig, ax = plt.subplots(2, 1)
ax[0].plot(
    posterior_mus, marker=".", linestyle="none", color="green", label="end point"
)
ax[0].axhline(prop_mean, linestyle="--", color="red")
ax[0].axhline(vis_mean, linestyle="--", color="blue")
ax[0].legend()
ax[1].plot(vis_sigs, marker=".", linestyle="none", color="blue", label="visual sigma")
ax[1].plot(prop_sigs, marker=".", linestyle="none", color="red", label="prop sigma")
ax[1].legend()

plt.show()
