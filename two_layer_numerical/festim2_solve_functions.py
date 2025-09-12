import festim as F
import numpy as np

my_model = F.HydrogenTransportProblemDiscontinuous()

barrier_thickness = 3e-6  # m
substrate_thickness = 3e-3  # m

points = np.concatenate((np.linspace(0, barrier_thickness, 100), np.linspace(barrier_thickness, substrate_thickness, 100)))
my_model.mesh = F.Mesh1D(vertices=points)

params = {
    "D_0_barrier": 1e-8,  # m2/s
    "E_D_barrier": 0.39,  # eV
    "S_0_barrier": 1e22,  # mol/m3Pa^0.5
    "E_S_barrier": 1.04,  # eV
    "K_r_0_barrier": 3.2e-15,  # m2/s Anderl 1992
    "E_K_r_barrier": 1.16,  # eV Anderl 1992
    "D_0_substrate": 4.1e-7,  # m2/s
    "E_D_substrate": 0.39,  # eV
    "S_0_substrate": 1.87e24,  # mol/m3Pa^0.5
    "E_S_substrate": 1.04,  # eV
    "K_r_0_substrate": 5.4e-19,  # m2/s
    "E_K_r_substrate": 15600 / 96491,  # converting to eV from F. WAELBROECK et al
}

tungsten = F.Material(
    D_0=params["D_0_barrier"],
    E_D=params["E_D_barrier"],
    K_S_0=params["S_0_barrier"],
    E_K_S=params["E_S_barrier"],
    solubility_law="sieverts",
)

stainless_steel = F.Material(
    D_0=params["D_0_substrate"],
    E_D=params["E_D_substrate"],
    K_S_0=params["S_0_substrate"],
    E_K_S=params["E_S_substrate"],
    solubility_law="sieverts",
)

barrier = F.VolumeSubdomain1D(id=1, material=tungsten, borders=[0, barrier_thickness])
substrate = F.VolumeSubdomain1D(id=2, material=stainless_steel, borders=[barrier_thickness, substrate_thickness])
left = F.SurfaceSubdomain1D(id=3, x=0)
right = F.SurfaceSubdomain1D(id=4, x=substrate_thickness)
interface = F.Interface(id=5, subdomains=[barrier, substrate], penalty_term=1e12)

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

my_model.temperature = 600

my_model.boundary_conditions = [
    F.SievertsBC(
        subdomain=left, S_0=tungsten.K_S_0, E_S=tungsten.E_K_S, pressure=100, species=H
    ),
    F.FixedConcentrationBC(subdomain=right, value=0, species=H),
]

my_model.settings = F.Settings(
    atol=1e-6,
    rtol=1e-10,
    transient=False,
)

my_model.exports = [
    F.VTXSpeciesExport(
        filename="results/two_layer_barrier.bp",
        field=H,
        subdomain=barrier,
    ),
    F.VTXSpeciesExport(
        filename="results/two_layer_substrate.bp",
        field=H,
        subdomain=substrate,
    ),
]

my_model.initialise()
my_model.run()