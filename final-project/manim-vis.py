from manim import *
import numpy as np


class SensoryIntegrationDecay(Scene):
    def construct(self):
        # Setup
        axes = Axes(
            x_range=[-5, 5, 1],
            y_range=[0, 0.5, 0.1],
            x_length=10,
            axis_config={"color": WHITE},
        )
        self.add(axes)

        # Parameters for visual and proprioceptive estimates
        mean_v, mean_p = 0, 0  # Means
        sigma_v, sigma_p = 1, 1.5  # Initial standard deviations
        sigma_v_inc = 0.1  # Increment in visual variance per step

        # Function to plot Gaussian
        def plot_gaussian(mu, sigma, color):
            return axes.plot(
                lambda x: 1
                / (sigma * np.sqrt(2 * np.pi))
                * np.exp(-0.5 * ((x - mu) / sigma) ** 2),
                color=color,
            )

        # Initial plots
        visual_estimate = plot_gaussian(mean_v, sigma_v, BLUE)
        proprio_estimate = plot_gaussian(mean_p, sigma_p, RED)
        self.play(Create(visual_estimate), Create(proprio_estimate))

        # Text for labels
        visual_label = Text("Visual Estimate", color=BLUE).to_edge(UP + LEFT)
        proprio_label = Text("Proprioceptive Estimate", color=RED).to_edge(UP + RIGHT)
        self.play(Write(visual_label), Write(proprio_label))

        # Updating visual variance and recalculating combined estimate
        for _ in range(5):  # Number of updates
            sigma_v += sigma_v_inc  # Increase visual variance

            # Update visual estimate plot
            new_visual_estimate = plot_gaussian(mean_v, sigma_v, BLUE)
            self.play(Transform(visual_estimate, new_visual_estimate), run_time=2)

        self.wait(2)
