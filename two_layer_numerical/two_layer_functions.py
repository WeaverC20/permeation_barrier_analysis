import festim as F
import numpy as np

def create_2_layer_model(params, barrier_thick, substrate_thick):
    """
    creates a 2 layer model

    Inputs:
    params: dictionary of parameters
    barrier_thick: thickness of the barrier layer (m)
    substrate_thick: thickness of the substrate layer (m)
    """
    model = F.Simulation()

    barrier = F.Material(
        id=5,
        D_0=params['D_0_barrier'],
        E_D=params['E_D_barrier'],
        S_0=params['S_0_barrier'],
        E_S=params['E_S_barrier'],
        borders=[0, barrier_thick]
    )

    substrate = F.Material(
        id=2,
        D_0=params['D_0_substrate'],
        E_D=params['E_D_substrate'],
        S_0=params['S_0_substrate'],
        E_S=params['E_S_substrate'],
        borders=[barrier_thick, substrate_thick + barrier_thick]
    )

    model.materials = [barrier, substrate]
    return model, barrier, substrate

def mesh_2_layer_model(model, barrier_thick, substrate_thick):
    """
    creates a mesh for a 2 layer model
    """
    vertices_barrier = np.linspace(0, barrier_thick, num=50)

    vertices_substrate = np.linspace(
        barrier_thick, substrate_thick + barrier_thick, num=50)

    vertices = np.concatenate([vertices_barrier, vertices_substrate])

    model.mesh = F.MeshFromVertices(vertices)

def set_2_layer_BCs(model, barrier, T=500, P_up = 100):
    """
    Sets the temperature of the mode  and boundary conditions
    """
    model.T = F.Temperature(T)

    left_bc = F.SievertsBC(
        surfaces=1,
        S_0=barrier.S_0,
        E_S=barrier.E_S,
        pressure=P_up
        )

    right_bc = F.DirichletBC(
        field="solute",
        surfaces=2,
        value=0
        )

    model.boundary_conditions = [left_bc, right_bc]

def set_2_layer_settings(model, results_folder="results", times_input = [100, 17000, 8e5]):
    derived_quantities_with_barrier = F.DerivedQuantities([F.HydrogenFlux(surface=2)], show_units=True)

    model.exports = [
        F.XDMFExport(
            field="solute",
            filename=results_folder + "/hydrogen_concentration.xdmf",
            checkpoint=False,  # needed in 1D
        ),
        F.TXTExport(
            field="solute",
            times=times_input,
            filename=results_folder + "/mobile_concentration.txt",
        ),
        derived_quantities_with_barrier
    ]

    model.settings = F.Settings(
        absolute_tolerance=1e0,
        relative_tolerance=1e-09,
        final_time=8e5,
        chemical_pot=True,
    )

    model.dt = F.Stepsize(
        initial_value=1,
        stepsize_change_ratio=1.1
    )

def compute_W_R(params, barrier_thick, substrate_thick, P_up, T):
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