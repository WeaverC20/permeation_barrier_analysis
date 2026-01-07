import festim as F
import numpy as np
import matplotlib.pyplot as plt


class StepsizeWithPostMilestoneControl(F.Stepsize):
    """
    Custom Stepsize that gradually reduces timestep before milestones and resets after.

    Behavior:
    - Before milestone: Gradually reduces timestep to initial_value
    - At milestone: Hits it exactly (handled by parent class)
    - After milestone: Starts at initial_value and grows naturally via adaptive stepping

    Args:
        pre_milestone_duration (float): Duration before milestone to start reducing timestep
        post_milestone_duration (float): Duration after milestone to enforce initial timestep
        *args, **kwargs: All other arguments passed to F.Stepsize
    """
    def __init__(self, *args, pre_milestone_duration=None, post_milestone_duration=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_milestone_duration = pre_milestone_duration
        self.post_milestone_duration = post_milestone_duration

    def modify_value(self, value, nb_iterations, t=None):
        # Call parent method first (handles normal adaptive stepping and hitting milestones)
        updated_value = super().modify_value(value, nb_iterations, t)

        if not self.milestones:
            return updated_value

        # Check if we're near any milestone
        for milestone in self.milestones:
            time_to_milestone = milestone - t
            time_since_milestone = t - milestone

            # BEFORE milestone: Gradually reduce to initial_value
            if self.pre_milestone_duration and 0 < time_to_milestone < self.pre_milestone_duration:
                # Linear interpolation: closer to milestone = closer to initial_value
                fraction = time_to_milestone / self.pre_milestone_duration
                # As we get closer (fraction → 0), max_allowed → initial_value
                max_allowed = self.initial_value + fraction * (updated_value - self.initial_value)
                updated_value = min(updated_value, max_allowed)
                break

            # AFTER milestone: Reset to initial_value and let it grow
            elif self.post_milestone_duration and 0 < time_since_milestone < self.post_milestone_duration:
                # Start at initial_value right after milestone
                # The adaptive stepping (growth_factor) will grow it from there
                updated_value = self.initial_value
                break

        return updated_value


def define_species(my_model, w_atom_density=6.3e28):
    H = F.Species("H")

    trapped_H1 = F.Species("trapped_H1", mobile=False)
    trapped_H2 = F.Species("trapped_H2", mobile=False)
    empty_trap1 = F.ImplicitSpecies(n=1.3e-3 * w_atom_density, others=[trapped_H1])
    empty_trap2 = F.ImplicitSpecies(n=4e-4 * w_atom_density, others=[trapped_H2])
    my_model.species = [H, trapped_H1, trapped_H2]

    return H, trapped_H1, trapped_H2, empty_trap1, empty_trap2

def define_BCs_and_initial_conditions(my_model, tungsten, substrate_thick, H, P_up):
    volume_subdomain = F.VolumeSubdomain1D(id=1, borders=(0, substrate_thick), material=tungsten)
    left_boundary = F.SurfaceSubdomain1D(id=2, x=0)
    right_boundary = F.SurfaceSubdomain1D(id=3, x=substrate_thick)

    my_model.subdomains = [
        left_boundary,
        volume_subdomain,
        right_boundary,
    ]

    left_bc = F.SievertsBC(
        subdomain=left_boundary,
        S_0=tungsten.K_S_0,
        E_S=tungsten.E_K_S,
        pressure=P_up,
        species=H,
        )

    right_bc = F.FixedConcentrationBC(
        species=H,
        subdomain=right_boundary,
        value=0
        )

    my_model.boundary_conditions = [left_bc, right_bc]

    return volume_subdomain, left_boundary, right_boundary

def define_trapping_reactions(my_model, H, trapped_H1, trapped_H2, empty_trap1, empty_trap2, volume_subdomain, atom_density):
    # Tungsten
    trapping_reaction_1 = F.Reaction(
        reactant=[H, empty_trap1],
        product=[trapped_H1],
        k_0=4.1e-7 / (1.1e-10**2 * 6 * atom_density),
        E_k=0.39,
        p_0=1e13,
        E_p=0.87,
        volume=volume_subdomain,
    )
    trapping_reaction_2 = F.Reaction(
        reactant=[H, empty_trap2],
        product=[trapped_H2],
        k_0=4.1e-7 / (1.1e-10**2 * 6 * atom_density),
        E_k=0.39,
        p_0=1e13,
        E_p=1.0,
        volume=volume_subdomain,
    )

    # # Stainless Steel
    # trapping_reaction_1 = F.Reaction(
    #     reactant=[H, empty_trap1],
    #     product=[trapped_H1],
    #     k_0=6e16 / (atom_density),   # m^3/s
    #     E_k=0.39,   # eV
    #     p_0=1e13,   # s^-1
    #     E_p=0.87,   # eV
    #     volume=volume_subdomain,
    # )

    # trapping_reaction_1 = F.Reaction(
    #     reactant=[H, empty_trap1],
    #     product=[trapped_H1],
    #     k_0=5e-29,
    #     E_k=0.39,   # eV
    #     p_0=3e1,   # s^-1
    #     E_p=0.87,    # eV
    #     volume=volume_subdomain,
    # )

    # trapping_reaction_2 = F.Reaction(
    #     reactant=[H, empty_trap2],
    #     product=[trapped_H2],
    #     k_0=2e-28,
    #     E_k=0.39,   # eV
    #     p_0=2.5e1,   # s^-1
    #     E_p=0.87,    # eV
    #     volume=volume_subdomain,
    # )

    # Tungsten
    my_model.reactions = [
        trapping_reaction_1,
        trapping_reaction_2,
    ]

def settings(my_model, final_time, growth_factor=1.05, cutback_factor=0.1, max_stepsize=200, milestones=[0], pre_milestone_duration=None, post_milestone_duration=None, initial_value=100):
    """
    Configure simulation settings with optional milestone timestep control.

    Args:
        my_model: The FESTIM model
        final_time: Final simulation time
        milestones: List of milestone times
        pre_milestone_duration: Duration before milestone to start reducing timestep (None = no control)
        post_milestone_duration: Duration after milestone to keep timestep at initial_value (None = no control)
        initial_value: Initial timestep value
    """
    my_model.settings = F.Settings(
        atol=1e0,
        rtol=1e-08,
        max_iterations=1000,
        final_time=final_time,
    )

    # Use custom Stepsize if milestone control is requested
    if pre_milestone_duration is not None or post_milestone_duration is not None:
        my_model.settings.stepsize = StepsizeWithPostMilestoneControl(
            initial_value=initial_value,
            growth_factor=growth_factor,
            cutback_factor=cutback_factor,
            max_stepsize=max_stepsize,
            target_nb_iterations=30,
            milestones=milestones,
            pre_milestone_duration=pre_milestone_duration,
            post_milestone_duration=post_milestone_duration,
        )
    else:
        # Use standard Stepsize
        my_model.settings.stepsize = F.Stepsize(
            initial_value=initial_value,
            growth_factor=growth_factor,
            cutback_factor=cutback_factor,
            max_stepsize=max_stepsize,
            target_nb_iterations=30,
            milestones=milestones
        )


def plot_1d_mesh(vertices):
    vertices = np.asarray(vertices)
    spacing = np.diff(vertices)

    fig, ax = plt.subplots(2, 1, figsize=(7, 4), sharex=True)

    # Plot mesh points
    ax[0].plot(vertices, np.zeros_like(vertices), 'o')
    ax[0].set_ylabel("Mesh points")
    ax[0].set_yticks([])
    ax[0].set_title("1D Mesh Distribution")

    # Plot spacing between points
    ax[1].plot(vertices[:-1], spacing, '-o')
    ax[1].set_ylabel("Δx")
    ax[1].set_xlabel("x")

    plt.tight_layout()
    plt.show()