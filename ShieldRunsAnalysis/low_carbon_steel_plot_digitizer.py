import numpy as np
import matplotlib.pyplot as plt


ironx = [1.231233595800525, 1.473753280839895, 1.7199475065616796, 1.8898950131233594]
irony = [2.403283216611514e-7, 9.378561332911526e-8, 3.6080748777038065e-8, 1.8460942515273254e-8]

steel_1020x = [1.2486876640419946, 1.48753280839895, 1.6510498687664041, 1.8402887139107609]
steel_1020y = [2.0253568259397192e-7, 7.153015466410034e-8, 3.556997757051428e-8, 1.5781286885029425e-8]

steel_1050x = [1.1715223097112861, 1.4158792650918635, 1.631758530183727, 1.8494750656167978]
steel_1050y = [1.7562326358966748e-7, 6.29160733250993e-8, 2.5625286558374804e-8, 1.0289254298577418e-8]

steel_1095x = [1.1549868766404199, 1.3561679790026246, 1.6363517060367454, 1.8219160104986876]
steel_1095y = [1.0216165638638841e-7, 4.468428352322411e-8, 1.4282320074118537e-8, 6.61353165407017e-9]

def convert_to_htm(x, y):
    """
    Converts from mol m^-1 s^-1 MPa^-1/2 to number atoms m^-1 s^-1 Pa^-1/2
    """
    x_converted = [val / 1e3 for val in x]
    y_converted = [val * 6.022e23 * (1/1e6)**0.5 for val in y]
    return (x_converted, y_converted)

steel_dataset = {
    "Iron": convert_to_htm(ironx, irony),
    "Steel 1020": convert_to_htm(steel_1020x, steel_1020y),
    "Steel 1050": convert_to_htm(steel_1050x, steel_1050y),
    "Steel 1095": convert_to_htm(steel_1095x, steel_1095y),
}

def plot_steel_htm(dataset, x_range=[0.00075, 0.00275]):
    for label, (x, y) in dataset.items():
        # simple Arrhenius (ln y vs 1/T) line fit
        m, b = np.polyfit(x, np.log(y), 1)
        xs = np.linspace(x_range[0], x_range[1], 200)
        # xs = np.linspace(min(x), max(x), 200)
        ys = np.exp(m*xs + b)
        plt.plot(xs, ys, linewidth=2, label=label)
    # plt.yscale("log")
    # plt.xlabel("1/T (K$^{-1}$)")
    # plt.ylabel("Permeability")
    # plt.grid(True, which="both", linestyle=":", linewidth=0.6)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

def plot_steel(dataset, x_range=[0.00075, 0.00275]):
    for label, (x, y) in dataset.items():
        # simple Arrhenius (ln y vs 1/T) line fit
        m, b = np.polyfit(x, np.log(y), 1)
        xs = np.linspace(x_range[0], x_range[1], 200)
        # xs = np.linspace(min(x), max(x), 200)
        ys = np.exp(m*xs + b)
        plt.plot(xs, ys, linewidth=2, label=f"{label} Sandia Report 2012")
    plt.yscale("log")
    plt.xlabel("1/T (K$^{-1}$)")
    plt.ylabel("Permeability")
    plt.grid(True, which="both", linestyle=":", linewidth=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_steel(steel_dataset)