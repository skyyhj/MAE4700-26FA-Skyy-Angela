import ast
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
# FEM functions (Dimension-Independent: 2D & 3D)
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


def assemble_global_stiffness(num_nodes, coords, connectivity, k_values, ndim):
    """
    Assemble the global stiffness matrix dynamically based on ndim.
    """
    ndof = ndim * num_nodes
    K = np.zeros((ndof, ndof), dtype=float)

    for e in range(len(connectivity)):
        nodes = connectivity[e]
        gdofs = element_dofs(nodes, ndim)
        
        x1 = coords[nodes[0]]
        x2 = coords[nodes[1]]
        ke = element_stiffness(x1, x2, k_values[e])

        K[np.ix_(gdofs, gdofs)] += ke
        
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
    Apply displacement boundary conditions and solve for nodal displacements.
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

    # Reaction vector calculation: R = K u - F
    full_residual = K @ u - F
    
    reactions = np.zeros_like(F)
    reactions[prescribed_dofs] = full_residual[prescribed_dofs]

    return u, reactions, free_dofs


def recover_element_forces(coords, u, connectivity, k_values, ndim):
    """
    Calculate the internal axial force in every element (dimension-independent).
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
    Main FEM workflow for 2D/3D trusses:
    1. Read input files
    2. Infer dimensionality (ndim) from nodal coordinates
    3. Assemble global stiffness matrix
    4. Build global force vector
    5. Apply BCs and solve system
    6. Calculate reactions and element internal forces
    7. Verify solution and write output files
    """
    # Read input files
    node_inputs = read_list_from_file("nodal coordinates.txt")
    connectivity_input = read_list_from_file("connectivity array.txt")
    k_values_input = read_list_from_file("element stiffnesses.txt")
    loads_input = read_list_from_file("external nodal forces.txt")
    prescribed_displacements = read_list_from_file("displacement BCs.txt")
    
    # Convert input data and infer dimensionality (ndim)
    coords = np.array(node_inputs, dtype=float)
    num_nodes, ndim = coords.shape

    if ndim not in (2, 3):
        raise ValueError("Only 2D and 3D trusses are supported.")

    connectivity = np.array(connectivity_input, dtype=int)
    k_values = np.array(k_values_input, dtype=float)
    
    F = build_force_vector(loads_input, num_nodes, ndim)
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

    # --------------------------------------------------------
    # Assemble global stiffness matrix
    # --------------------------------------------------------
    K = assemble_global_stiffness(num_nodes, coords, connectivity, k_values, ndim)

    # --------------------------------------------------------
    # Solve system
    # --------------------------------------------------------
    u, reactions, free_dofs = solve_system(K, F, prescribed_displacements, ndim)

    # --------------------------------------------------------
    # Recover element internal forces
    # --------------------------------------------------------
    internal_forces = recover_element_forces(coords, u, connectivity, k_values, ndim)

    # --------------------------------------------------------
    # Verification
    # --------------------------------------------------------
    check_solution(K, F, u, reactions, free_dofs)

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
