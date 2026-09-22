import ast
from pathlib import Path
import numpy as np


# ------------------------------------------------------------
# Input / output utilities
# ------------------------------------------------------------
def read_list_from_file(filename):
    """
    Read a text file containing a raw Python list.

    Returns:
        Python list
    """
    with open(filename, "r") as file:
        return ast.literal_eval(file.read().strip())


def write_list_to_file(filename, values):
    """
    Write a Python list to a text file.
    """
    with open(filename, "w") as file:
        file.write(repr(values))


# ------------------------------------------------------------
# FEM functions (Dimension-Independent: 2D & 3D + Thermal)
# ------------------------------------------------------------
def node_dofs(node, ndim):
    """
    Get global degrees of freedom for a single node.
    """
    first = node * ndim
    return np.arange(first, first + ndim, dtype=int)


def element_dofs(nodes, ndim):
    """
    Get global degrees of freedom for a two-node element.
    """
    i, j = nodes
    return np.concatenate((node_dofs(i, ndim), node_dofs(j, ndim)))


def element_geometry(x1, x2):
    """
    Get element length and direction vector n (2D or 3D).
    """
    dx = np.asarray(x2, dtype=float) - np.asarray(x1, dtype=float)
    L = np.linalg.norm(dx)

    if L <= 0.0:
        raise ValueError("Truss element has zero length")

    n = dx / L
    return L, n


def element_operator(x1, x2):
    """
    Compute element length, direction vector, and axial projection matrix B.
    B = [-n^T, n^T]
    """
    L, n = element_geometry(x1, x2)
    B = np.concatenate((-n, n))
    return L, n, B


def element_stiffness(x1, x2, k):
    """
    Return the element stiffness matrix in global coordinates using ke = k * outer(B, B).
    """
    L, n, B = element_operator(x1, x2)
    ke = k * np.outer(B, B)
    return ke


def element_thermal_force(x1, x2, k, alpha, dT):
    """
    Compute element thermal equivalent nodal force vector: f_th = k * L * alpha * dT * B
    """
    L, n, B = element_operator(x1, x2)
    return k * L * alpha * dT * B


def assemble_global_system(num_nodes, coords, connectivity, k_values, ndim, thermal=False, alpha=None, dT=None):
    """
    Assemble the global stiffness matrix K and global force vector F (including thermal loads if applicable).
    """
    ndof = ndim * num_nodes
    K = np.zeros((ndof, ndof), dtype=float)
    return K


def build_force_vector(loads_input, num_nodes, ndim):
    """
    Convert the list of nodal loads into a NumPy force vector and validate dimensions.
    """
    loads_array = np.array(loads_input, dtype=float)
    if loads_array.shape != (num_nodes, ndim):
        raise ValueError("Force file dimension does not match coordinates")

    F = loads_array.reshape(-1)
    return F


def solve_system(K, F, prescribed_displacements, ndim):
    """
    Apply displacement boundary conditions and solve for nodal displacements and reactions.
    """
    num_dofs = len(F)

    prescribed_dofs = []
    free_dofs = []

    u = np.zeros(num_dofs, dtype=float)

    for node, pair in enumerate(prescribed_displacements):
        if len(pair) != ndim:
            raise ValueError("BC dimension does not match coordinates")
        for direction, value in enumerate(pair):
            gdof = node * ndim + direction
            if value is None:
                free_dofs.append(gdof)
            else:
                prescribed_dofs.append(gdof)
                u[gdof] = float(value)

    if len(free_dofs) == 0:
        raise ValueError("There are no free degrees of freedom.")

    K_ff = K[np.ix_(free_dofs, free_dofs)]
    K_fe = K[np.ix_(free_dofs, prescribed_dofs)]

    F_f = F[free_dofs]
    u_e = u[prescribed_dofs]

    rhs = F_f - K_fe @ u_e

    try:
        u_f = np.linalg.solve(K_ff, rhs)
    except np.linalg.LinAlgError:
        raise ValueError(
            "The free stiffness matrix is singular. "
            "The structure may have an unconstrained rigid-body motion "
            "or insufficient displacement boundary conditions."
        )

    u[free_dofs] = u_f

    # Reaction vector calculation: R = K u - F (where F includes thermal loads)
    full_residual = K @ u - F

    reactions = np.zeros_like(F)
    reactions[prescribed_dofs] = full_residual[prescribed_dofs]

    return u, reactions, free_dofs


def recover_element_forces(coords, u, connectivity, k_values, ndim, alpha=None, dT=None):
    """
    Calculate the internal axial force in every element, accounting for thermal strains if provided.
    """
    internal_forces = []

    for e in range(len(connectivity)):
        nodes = connectivity[e]
        gdofs = element_dofs(nodes, ndim)
        element_u = u[gdofs]

        i = nodes[0]
        j = nodes[1]

        x1 = coords[i]
        x2 = coords[j]

        L, n, B = element_operator(x1, x2)
        axial_extension = B @ element_u

        if alpha is not None and dT is not None:
            thermal_extension = L * alpha[e] * dT[e]
            force = k_values[e] * (axial_extension - thermal_extension)
        else:
            force = k_values[e] * axial_extension

        internal_forces.append(force)

    return np.array(internal_forces, dtype=float)


# ------------------------------------------------------------
# Verification
# ------------------------------------------------------------

def check_solution(K, F, u, reactions, free_dofs):
    """
    Perform numerical and mechanical checks.
    """
    tolerance = 1e-10

    symmetry_check = np.allclose(K, K.T, atol=tolerance)
    residual = K @ u - F
    free_residual_check = np.allclose(residual[free_dofs], 0.0, atol=tolerance)
    equilibrium_check = np.isclose(np.sum(F) + np.sum(reactions), 0.0, atol=tolerance)

    print("\nVerification checks:")
    print(f"  Global stiffness matrix symmetric: {symmetry_check}")
    print(f"  Free-DOF residual approximately zero: {free_residual_check}")
    print(f"  Global force equilibrium satisfied: {equilibrium_check}")

    if not symmetry_check:
        print("WARNING: Global stiffness matrix is not symmetric.")
    if not free_residual_check:
        print("WARNING: Residual is not zero at all free DOFs.")
    if not equilibrium_check:
        print("WARNING: Global force equilibrium is not satisfied.")

    return symmetry_check and free_residual_check and equilibrium_check


# ------------------------------------------------------------
# Main program
# ------------------------------------------------------------

def main():
    """
    Main FEM workflow for 2D/3D trusses with optional thermal effects:
    1. Read standard input files and check for optional thermal files
    2. Infer dimensionality (ndim) from nodal coordinates
    3. Assemble global stiffness matrix and global force vector (including thermal loads)
    4. Apply BCs and solve system
    5. Calculate reactions and element internal forces
    6. Verify solution and write output files
    """
    # Read standard input files
    node_inputs = read_list_from_file("nodal coordinates.txt")
    connectivity_input = read_list_from_file("connectivity array.txt")
    k_values_input = read_list_from_file("element stiffnesses.txt")
    loads_input = read_list_from_file("external nodal forces.txt")
    prescribed_displacements = read_list_from_file("displacement BCs.txt")

    # Check for optional thermal input files
    alpha_file = Path("thermal expansion coeff.txt")
    dT_file = Path("temperature change.txt")
    thermal = alpha_file.exists() or dT_file.exists()

    if thermal and not (alpha_file.exists() and dT_file.exists()):
        raise ValueError("Both thermal input files are required if thermal analysis is requested.")

    if thermal:
        alpha_input = read_list_from_file("thermal expansion coeff.txt")
        dT_input = read_list_from_file("temperature change.txt")
        alpha = np.array(alpha_input, dtype=float)
        dT = np.array(dT_input, dtype=float)
    else:
        alpha = None
        dT = None

    # Convert input data and infer dimensionality (ndim)
    coords = np.array(node_inputs, dtype=float)
    num_nodes, ndim = coords.shape

    if ndim not in (2, 3):
        raise ValueError("Only 2D and 3D trusses are supported.")

    connectivity = np.array(connectivity_input, dtype=int)
    k_values = np.array(k_values_input, dtype=float)

    F_external = build_force_vector(loads_input, num_nodes, ndim)
    num_elements = len(connectivity)

    # --------------------------------------------------------
    # Input validation
    # --------------------------------------------------------
    if connectivity.shape != (num_elements, 2):
        raise ValueError("Connectivity array must contain two nodes per element.")

    if len(k_values) != num_elements:
        raise ValueError("Number of element stiffnesses must equal number of elements.")

    if len(prescribed_displacements) != num_nodes:
        raise ValueError("Number of displacement BCs must equal number of nodes.")

    if np.any(connectivity < 0) or np.any(connectivity >= num_nodes):
        raise ValueError("Connectivity contains an invalid node number.")

    if thermal:
        if len(alpha) != num_elements or len(dT) != num_elements:
            raise ValueError("Thermal input lists must contain one value per element.")

    # --------------------------------------------------------
    # Assemble global stiffness matrix and force vector
    # --------------------------------------------------------
    ndof = ndim * num_nodes
    K = np.zeros((ndof, ndof), dtype=float)
    F = F_external.copy()

    for e in range(num_elements):
        nodes = connectivity[e]
        gdofs = element_dofs(nodes, ndim)

        x1 = coords[nodes[0]]
        x2 = coords[nodes[1]]
        ke = element_stiffness(x1, x2, k_values[e])

        K[np.ix_(gdofs, gdofs)] += ke

        if thermal:
            fth = element_thermal_force(x1, x2, k_values[e], alpha[e], dT[e])
            F[gdofs] += fth

    # --------------------------------------------------------
    # Solve system
    # --------------------------------------------------------
    u, reactions, free_dofs = solve_system(K, F, prescribed_displacements, ndim)

    # --------------------------------------------------------
    # Recover element internal forces
    # --------------------------------------------------------
    internal_forces = recover_element_forces(coords, u, connectivity, k_values, ndim, alpha, dT)

    # --------------------------------------------------------
    # Verification
    # --------------------------------------------------------
    check_solution(K, F, u, reactions, free_dofs)

    # --------------------------------------------------------
    # Strain & Stress Reporting
    # --------------------------------------------------------
    if thermal:
        # Explicit material properties given for Question 4(c)
        E_steel = 200e9  # Pa (200 GPa)
        A_steel = 1e-4   # m^2 (1 cm^2)
        element_idx = 0  # Change to the specific non-zero bar you want to report

        nodes = connectivity[element_idx]
        gdofs = element_dofs(nodes, ndim)
        element_u = u[gdofs]

        x1 = coords[nodes[0]]
        x2 = coords[nodes[1]]
        L, n, B = element_operator(x1, x2)

        # Calculate strains and stress
        eps_tot = (B @ element_u) / L
        eps_th = alpha[element_idx] * dT[element_idx]
        eps_mech = eps_tot - eps_th
        thermal_extension = L * alpha[element_idx] * dT[element_idx]
        force = k_values[element_idx] * ((B @ element_u) - thermal_extension)
        stress = force / A_steel

        print(f"\n--- Strain & Stress Report (Element {element_idx}) ---")
        print(f"  Total Strain (eps_tot):       {eps_tot:.6e}")
        print(f"  Thermal Strain (eps_th):      {eps_th:.6e}")
        print(f"  Mechanical Strain (eps_mech): {eps_mech:.6e}")
        print(f"  Axial Stress (sigma):         {stress:.4f} Pa ({stress / 1e6:.2f} MPa)")
    # --------------------------------------------------------
    # Write required output files
    # --------------------------------------------------------
    nodal_displacements = u.reshape((num_nodes, ndim))
    write_list_to_file("nodal displacements.txt", nodal_displacements.tolist())

    reaction_pairs = reactions.reshape((num_nodes, ndim))
    write_list_to_file("reaction forces.txt", reaction_pairs.tolist())

    write_list_to_file("internal forces.txt", internal_forces.tolist())

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------
    print("\nGlobal stiffness matrix K:")
    print(K)

    print("\nNodal displacements:")
    print(nodal_displacements.tolist())

    print("\nReaction forces:")
    print(reaction_pairs.tolist())

    print("\nInternal element forces:")
    print(internal_forces.tolist())

    print("\nOutput files created successfully.")


if __name__ == "__main__":
    main()
