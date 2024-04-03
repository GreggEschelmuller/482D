from models import CueCombinationModel
import matplotlib.pyplot as plt


cue_1 = (6, 1)
cue_2 = (2, 1)
cue_3 = (10, 1)

my_model = CueCombinationModel()

my_model.add_cues([cue_1, cue_2, cue_3])
fig, ax = my_model.plot_posterior_and_cues()
plt.show()

print(my_model.get_sigma())

my_model.add_cues([(4, 0.5)])
fig, ax = my_model.plot_posterior_and_cues()
plt.show()
