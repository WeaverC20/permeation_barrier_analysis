import festim as F
import numpy as np
import matplotlib.pyplot as plt

def create_mesh(my_model, barrier_thickness, substrate_thickness):
    """
    creates a 1D mesh for the two-layer model with 100 elements in each layer
    """
    points = np.concatenate((np.linspace(0, barrier_thickness, 100), np.linspace(barrier_thickness, barrier_thickness + substrate_thickness, 100)))
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
    substrate = F.VolumeSubdomain1D(id=2, material=mat2, borders=[barrier_thickness, barrier_thickness + substrate_thickness])
    left = F.SurfaceSubdomain1D(id=3, x=0)
    right = F.SurfaceSubdomain1D(id=4, x=barrier_thickness + substrate_thickness)
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

def compute_W_R(params, barrier_thick, substrate_thick, T, P_up):
    """
    computes the dimensionless quantities W and R from the given parameters
    """
    k_B = 8.617333262145e-5  # eV/K

    D_barrier = params['D_0_barrier'] * np.exp(-params['E_D_barrier']/(k_B * T))
    D_substrate = params['D_0_substrate'] * np.exp(-params['E_D_substrate']/(k_B * T))

    S_barrier = params['S_0_barrier'] * np.exp(-params['E_S_barrier']/(2 * k_B * T))
    S_substrate = params['S_0_substrate'] * np.exp(-params['E_S_substrate']/(2 * k_B * T))

    K_r_barrier = params['K_r_0_barrier'] * np.exp(-params['E_K_r_barrier']/(k_B * T))
    K_d_barrier = K_r_barrier * S_barrier ** 2

    K_r_substrate = params['K_r_0_substrate'] * np.exp(-params['E_K_r_substrate']/(k_B * T))
    K_d_substrate = K_r_substrate * S_substrate ** 2

    W = K_d_barrier * P_up ** 0.5 * (
        barrier_thick / (D_barrier * S_barrier) +
        substrate_thick / (D_substrate * S_substrate)
        )
    
    R = K_d_substrate / K_d_barrier

    return W, R

def plot_concentration_profile(my_model, barrier_export, substrate_export, results_folder="results"):
    """
    plots the concentration profile from the exports
    """
    c_array = np.concatenate((barrier_export.data[0], substrate_export.data[0]))
    points = np.concatenate((my_model.mesh.vertices[:100], my_model.mesh.vertices[99:]))

    plt.plot(points, c_array)
    plt.xlabel("x (m)")
    plt.ylabel("Mobile concentration (H/m3)")
    plt.title("Concentration profile")
    plt.show()