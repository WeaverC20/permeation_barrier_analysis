import festim as F
import numpy as np

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

def define_trapping_reactions(my_model, H, trapped_H1, trapped_H2, empty_trap1, empty_trap2, volume_subdomain, w_atom_density):
    trapping_reaction_1 = F.Reaction(
        reactant=[H, empty_trap1],
        product=[trapped_H1],
        k_0=4.1e-7 / (1.1e-10**2 * 6 * w_atom_density),
        E_k=0.39,
        p_0=1e13,
        E_p=0.87,
        volume=volume_subdomain,
    )
    trapping_reaction_2 = F.Reaction(
        reactant=[H, empty_trap2],
        product=[trapped_H2],
        k_0=4.1e-7 / (1.1e-10**2 * 6 * w_atom_density),
        E_k=0.39,
        p_0=1e13,
        E_p=1.0,
        volume=volume_subdomain,
    )

    my_model.reactions = [
        trapping_reaction_1,
        trapping_reaction_2,
    ]

def settings(my_model, final_time, milestones=[0]):
    my_model.settings = F.Settings(
        atol=1e0,
        rtol=1e-08,
        final_time=final_time,
    )

    my_model.settings.stepsize = F.Stepsize(
        initial_value=5, growth_factor=2, cutback_factor=0.0001, max_stepsize=200, target_nb_iterations=5,
        milestones=milestones
    )
