from manim import *
import numpy as np


class SensoryIntegrationDecay_(Scene):
    def construct(self):
        # Setup
        axes = Axes(
            x_range=[-10, 10, 1],
            y_range=[0, 0.5, 0.1],
            x_length=10,
            axis_config={"color": WHITE},
        )
        self.add(axes)

        # Parameters for visual and proprioceptive estimates
        mean_v, mean_p = 1, -4  # Means
        sigma_v, sigma_p = 1, 1.5  # Initial standard deviations

        sigma_v_inc = 0.25  # Increment in visual variance per step
        sigma_p_inc = 0.05

        def calc_post(bias_p, bias_v, sigma_v, sigma_p):
            j_v = 1 / sigma_v**2
            j_p = 1 / sigma_p**2
            mean_pos = (j_v * bias_v + j_p * bias_p) / (j_v + j_p)
            sigma_pos = np.sqrt(1 / (j_v + j_p))
            return mean_pos, sigma_pos

        # Function to plot Gaussian
        def plot_gaussian(mu, sigma, color):
            return axes.plot(
                lambda x: 1
                / (sigma * np.sqrt(2 * np.pi))
                * np.exp(-0.5 * ((x - mu) / sigma) ** 2),
                color=color,
            )

        # Initial plots
        mean_pos, sigma_pos = calc_post(mean_p, mean_v, sigma_v, sigma_p)
        visual_estimate = plot_gaussian(mean_v, sigma_v, BLUE)
        proprio_estimate = plot_gaussian(mean_p, sigma_p, RED)
        posterior_estimate = plot_gaussian(mean_pos, sigma_pos, GREEN)
        # self.play(
        #     Create(visual_estimate),
        #     Create(proprio_estimate),
        #     Create(posterior_estimate),
        # )
        self.wait(1)

        visual_label = (
            Text("Visual Estimate", color=BLUE)
            .scale(0.3)
            .to_edge(UP + LEFT)
            .shift(RIGHT * 3)
            .shift(DOWN * 0.5)
        )
        proprio_label = (
            Text("Proprioceptive Estimate", color=RED)
            .scale(0.3)
            .next_to(visual_label, DOWN)
        )
        posterior_label = (
            Text("Perceived Position", color=GREEN)
            .scale(0.3)
            .next_to(proprio_label, DOWN)
        )

        # self.play(Write(visual_label), Write(proprio_label), Write(posterior_label))

        self.play(Create(visual_estimate), Write(visual_label), run_time=1)
        self.play(Create(proprio_estimate), Write(proprio_label), run_time=1)
        self.play(Create(posterior_estimate), Write(posterior_label), run_time=1)

        # Updating visual variance and recalculating combined estimate
        for _ in range(20):  # Number of updates
            sigma_v += sigma_v_inc  # Increase visual variance
            sigma_p += sigma_p_inc

            # Update visual estimate plot
            new_visual_estimate = plot_gaussian(mean_v, sigma_v, BLUE)
            new_prop_estimate = plot_gaussian(mean_p, sigma_p, RED)
            mean_pos, sigma_pos = calc_post(mean_p, mean_v, sigma_v, sigma_p)
            new_posterior = plot_gaussian(mean_pos, sigma_pos, GREEN)
            self.play(
                Transform(visual_estimate, new_visual_estimate),
                Transform(proprio_estimate, new_prop_estimate),
                Transform(posterior_estimate, new_posterior),
                run_time=1,
            )

        self.wait(1)
