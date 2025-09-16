import festim as F
import numpy as np

def create_mesh(my_model, barrier_thickness, substrate_thickness):
    """
    creates a 1D mesh for the two-layer model with 100 elements in each layer
    """
    points = np.concatenate((np.linspace(0, barrier_thickness, 100), np.linspace(barrier_thickness, substrate_thickness, 100)))
    my_model.mesh = F.Mesh1D(vertices=points)

def assign_materials_and_domains(my_model, params, barrier_thickness, substrate_thickness, T=500, P_up=100):
    """
    assigns materials and domains to the model
    """
    mat1 = F.Material(
        D_0=params["D_0_barrier"],
        E_D=params["E_D_barrier"],
        K_S_0=params["S_0_barrier"],
        E_K_S=params["E_S_barrier"],
        solubility_law="sieverts",
    )
    mat2 = F.Material(
        D_0=params["D_0_substrate"],
        E_D=params["E_D_substrate"],
        K_S_0=params["S_0_substrate"],
        E_K_S=params["E_S_substrate"],
        solubility_law="sieverts",
    )

    barrier = F.VolumeSubdomain1D(id=1, material=mat1, borders=[0, barrier_thickness])
    substrate = F.VolumeSubdomain1D(id=2, material=mat2, borders=[barrier_thickness, substrate_thickness])
    left = F.SurfaceSubdomain1D(id=3, x=0)
    right = F.SurfaceSubdomain1D(id=4, x=substrate_thickness)
    interface = F.Interface(id=5, subdomains=[barrier, substrate], penalty_term=1e17)

    my_model.interfaces = [interface]
    my_model.subdomains = [
        barrier,
        substrate,
        left,
        right,
    ]

    H = F.Species(name="H", mobile=True, subdomains=[barrier, substrate])
    my_model.species = [H]

    my_model.surface_to_volume = {
        left: barrier,
        right: substrate,
    }

    my_model.temperature = T

    sieverts_bc = F.SievertsBC(
        subdomain=left, S_0=mat1.K_S_0, E_S=mat1.E_K_S, pressure=P_up, species=H
    )


    my_model.boundary_conditions = [
        sieverts_bc,
        F.FixedConcentrationBC(subdomain=right, value=0, species=H),
    ]

    return barrier, substrate, H

def set_exports(my_model, barrier, substrate, H, results_folder="results"):
    """
    sets the exports for the model
    """
    barrier_export = F.Profile1DExport(
        field=H,
        subdomain=barrier,
    )
    substrate_export = F.Profile1DExport(
        field=H,
        subdomain=substrate,
    )

    my_model.exports = [
        F.VTXSpeciesExport(
            filename=f"{results_folder}/two_layer_barrier.bp",
            field=H,
            subdomain=barrier,
        ),
        F.VTXSpeciesExport(
            filename=f"{results_folder}/two_layer_substrate.bp",
            field=H,
            subdomain=substrate,
        ),
        barrier_export,
        substrate_export,
    ]

    return barrier_export, substrate_export
