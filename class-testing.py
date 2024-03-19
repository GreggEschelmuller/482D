from models import CueCombinationModel

cue_1 = (2, 0.5)
cue_2 = (5, 1)
cue_3 = (3, 2)

my_model = CueCombinationModel()

my_model.add_cues([cue_1, cue_2, cue_3])
fig, ax = my_model.plot_posterior_and_cues()

fig.show()
input("press enter when ready")
