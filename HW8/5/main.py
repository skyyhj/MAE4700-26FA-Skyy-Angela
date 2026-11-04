
import ast
import numpy as np


# ------------------------------------------------------------
# Input / output utilities
# ------------------------------------------------------------

def read_list_from_file(filename):
    """
    Read a text file containing a raw Python list.
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
# FEM functions: 2D Plane Frame
# ------------------------------------------------------------

def node_dofs(node):
    """
    Return global DOFs for a node.

    Each node has three DOFs:
        ux, uy, theta
    """
    first = 3 * node
    return np.arange(first, first + 3, dtype=int)


def element_dofs(nodes):
    """
    Return the six global DOFs for a two-node frame element.
    """
    i, j = nodes

    return np.concatenate((
        node_dofs(i),
        node_dofs(j)
    ))


def element_geometry(x1, x2):
    """
    Calculate element length and direction cosines.
    """
    dx = np.asarray(x2, dtype=float) - np.asarray(x1, dtype=float)

    L = np.linalg.norm(dx)

    if L <= 0.0:
        raise ValueError("Frame element has zero length.")

    c = dx[0] / L
    s = dx[1] / L

    return L, c, s


def transformation_matrix(c, s):
    """
    Transform global element displacements into local coordinates.

    d_local = T @ d_global
    """

    T = np.array([
        [ c,  s, 0,  0,  0, 0],
        [-s,  c, 0,  0,  0, 0],
        [ 0,  0, 1,  0,  0, 0],
        [ 0,  0, 0,  c,  s, 0],
        [ 0,  0, 0, -s,  c, 0],
        [ 0,  0, 0,  0,  0, 1]
    ], dtype=float)

    return T


def local_element_stiffness(L, k, EI):
    """
    Construct the 6x6 local stiffness matrix.

    k = EA/L
    EI = flexural rigidity

    Local DOF order:
    [u1, v1, theta1, u2, v2, theta2]
    """

    a = 12.0 * EI / L**3
    b = 6.0 * EI / L**2
    c = 4.0 * EI / L
    d = 2.0 * EI / L

    ke = np.array([
        [ k,  0,  0, -k,  0,  0],
        [ 0,  a,  b,  0, -a,  b],
        [ 0,  b,  c,  0, -b,  d],
        [-k,  0,  0,  k,  0,  0],
        [ 0, -a, -b,  0,  a, -b],
        [ 0,  b,  d,  0, -b,  c]
    ], dtype=float)

    return ke


def consistent_load_vector(L, p):
    """
    Consistent local nodal load vector for a uniform
    transverse distributed load p.

    Positive p acts in the positive local y direction.

    Local order:
    [Fx1, Fy1, M1, Fx2, Fy2, M2]
    """

    fp = np.array([
        0.0,
        p * L / 2.0,
        p * L**2 / 12.0,
        0.0,
        p * L / 2.0,
        -p * L**2 / 12.0
    ], dtype=float)

    return fp


def element_matrices(x1, x2, k, EI, p):
    """
    Calculate local and global element stiffness matrices,
    transformation matrix, and consistent load vectors.
    """

    L, c, s = element_geometry(x1, x2)

    T = transformation_matrix(c, s)

    ke_local = local_element_stiffness(L, k, EI)

    fp_local = consistent_load_vector(L, p)

    # Transform local quantities into global coordinates
    ke_global = T.T @ ke_local @ T

    fp_global = T.T @ fp_local

    return ke_local, ke_global, T, fp_local, fp_global


# ------------------------------------------------------------
# Global assembly
# ------------------------------------------------------------

def assemble_global_system(
    num_nodes,
    coords,
    connectivity,
    k_values,
    EI_values,
    distributed_loads
):
    """
    Assemble the global stiffness matrix and
    global consistent distributed-load vector.
    """

    ndof = 3 * num_nodes

    K = np.zeros((ndof, ndof), dtype=float)
    Fp = np.zeros(ndof, dtype=float)

    for e in range(len(connectivity)):

        nodes = connectivity[e]
        gdofs = element_dofs(nodes)

        x1 = coords[nodes[0]]
        x2 = coords[nodes[1]]

        ke_local, ke_global, T, fp_local, fp_global = (
            element_matrices(
                x1,
                x2,
                k_values[e],
                EI_values[e],
                distributed_loads[e]
            )
        )

        # Assemble stiffness
        K[np.ix_(gdofs, gdofs)] += ke_global

        # Assemble consistent distributed loads
        Fp[gdofs] += fp_global

    return K, Fp


# ------------------------------------------------------------
# External nodal force vector
# ------------------------------------------------------------

def build_force_vector(loads_input, num_nodes):
    """
    Convert nodal loads into a global force vector.

    Each node must have:
    [Fx, Fy, M]
    """

    loads_array = np.array(loads_input, dtype=float)

    if loads_array.shape != (num_nodes, 3):
        raise ValueError(
            "Force file must contain [Fx, Fy, M] for each node."
        )

    return loads_array.reshape(-1)


# ------------------------------------------------------------
# Apply boundary conditions and solve
# ------------------------------------------------------------

def solve_system(K, F, prescribed_displacements):
    """
    Apply displacement boundary conditions and solve.

    None = free DOF
    Number = prescribed displacement or rotation

    Supports nonzero prescribed displacements.
    """

    num_dofs = len(F)

    prescribed_dofs = []
    free_dofs = []

    u = np.zeros(num_dofs, dtype=float)

    for node, values in enumerate(prescribed_displacements):

        if len(values) != 3:
            raise ValueError(
                "Each node must have three displacement BC entries."
            )

        for direction, value in enumerate(values):

            gdof = 3 * node + direction

            if value is None:
                free_dofs.append(gdof)

            else:
                prescribed_dofs.append(gdof)
                u[gdof] = float(value)

    # Solve only if free DOFs exist
    if len(free_dofs) > 0:

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
                "Check the supports and element connectivity."
            )

        u[free_dofs] = u_f

    # Reactions from global equilibrium
    full_residual = K @ u - F

    reactions = np.zeros(num_dofs, dtype=float)

    reactions[prescribed_dofs] = (
        full_residual[prescribed_dofs]
    )

    return u, reactions, free_dofs


# ------------------------------------------------------------
# Recover local element end forces
# ------------------------------------------------------------

def recover_element_forces(
    coords,
    u,
    connectivity,
    k_values,
    EI_values,
    distributed_loads
):
    """
    Recover six local end forces for every element.

    f_internal = ke_local @ d_local - fp_local

    Output order:
    [N1, V1, M1, N2, V2, M2]
    """

    internal_forces = []

    for e in range(len(connectivity)):

        nodes = connectivity[e]
        gdofs = element_dofs(nodes)

        x1 = coords[nodes[0]]
        x2 = coords[nodes[1]]

        ke_local, ke_global, T, fp_local, fp_global = (
            element_matrices(
                x1,
                x2,
                k_values[e],
                EI_values[e],
                distributed_loads[e]
            )
        )

        # Global element displacement vector
        d_global = u[gdofs]

        # Convert to local displacements
        d_local = T @ d_global

        # Recover local end forces
        f_internal = ke_local @ d_local - fp_local

        internal_forces.append(f_internal)

    return np.array(internal_forces, dtype=float)


# ------------------------------------------------------------
# Verification
# ------------------------------------------------------------

def check_solution(
    K,
    F,
    u,
    reactions,
    free_dofs,
    coords,
    connectivity,
    internal_forces
):
    """
    Verify stiffness symmetry, free-DOF equilibrium,
    and overall force and moment equilibrium.
    """

    tolerance = 1e-8

    symmetry_check = np.allclose(
        K, K.T, rtol=1e-10, atol=tolerance
    )

    residual = K @ u - F

    free_residual_check = np.allclose(
        residual[free_dofs],
        0.0,
        rtol=1e-8,
        atol=tolerance
    )

    # Force equilibrium from nodal applied loads and reactions
    total_forces = F + reactions

    Fx = np.sum(total_forces[0::3])
    Fy = np.sum(total_forces[1::3])

    # Sum moments about the global origin
    M = np.sum(total_forces[2::3])

    for i, (x, y) in enumerate(coords):
        M += x * total_forces[3*i + 1]
        M -= y * total_forces[3*i]

    equilibrium_check = np.allclose(
        [Fx, Fy, M],
        [0.0, 0.0, 0.0],
        rtol=1e-8,
        atol=tolerance
    )

    print("\nVerification checks:")
    print(f"  Stiffness matrix symmetric: {symmetry_check}")
    print(f"  Free-DOF equilibrium: {free_residual_check}")
    print(f"  Global equilibrium: {equilibrium_check}")

    return (
        symmetry_check
        and free_residual_check
        and equilibrium_check
    )


# ------------------------------------------------------------
# Main program
# ------------------------------------------------------------

def main():
    """
    Main FEM workflow for 2D plane frames.

    1. Read input files
    2. Validate inputs
    3. Assemble global stiffness and load vectors
    4. Apply boundary conditions
    5. Solve for nodal displacements
    6. Calculate support reactions
    7. Recover local element end forces
    8. Verify and write output files
    """

    # --------------------------------------------------------
    # Read input files
    # --------------------------------------------------------

    node_inputs = read_list_from_file(
        "nodal coordinates.txt"
    )

    connectivity_input = read_list_from_file(
        "connectivity array.txt"
    )

    k_values_input = read_list_from_file(
        "element stiffnesses.txt"
    )

    EI_values_input = read_list_from_file(
        "element flexural rigidities.txt"
    )

    distributed_loads_input = read_list_from_file(
        "distributed loads.txt"
    )

    loads_input = read_list_from_file(
        "external nodal forces.txt"
    )

    prescribed_displacements = read_list_from_file(
        "displacement BCs.txt"
    )

    # --------------------------------------------------------
    # Convert and validate input data
    # --------------------------------------------------------

    coords = np.array(node_inputs, dtype=float)

    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(
            "Nodal coordinates must be [x, y] pairs."
        )

    num_nodes = len(coords)

    connectivity = np.array(
        connectivity_input, dtype=int
    )

    if connectivity.ndim != 2 or connectivity.shape[1] != 2:
        raise ValueError(
            "Connectivity must contain two nodes per element."
        )

    num_elements = len(connectivity)

    k_values = np.array(
        k_values_input, dtype=float
    )

    EI_values = np.array(
        EI_values_input, dtype=float
    )

    distributed_loads = np.array(
        distributed_loads_input, dtype=float
    )

    if len(k_values) != num_elements:
        raise ValueError(
            "Incorrect number of element axial stiffnesses."
        )

    if len(EI_values) != num_elements:
        raise ValueError(
            "Incorrect number of flexural rigidities."
        )

    if len(distributed_loads) != num_elements:
        raise ValueError(
            "Incorrect number of distributed loads."
        )

    if len(prescribed_displacements) != num_nodes:
        raise ValueError(
            "Incorrect number of displacement BC entries."
        )

    if np.any(connectivity < 0) or np.any(
        connectivity >= num_nodes
    ):
        raise ValueError(
            "Connectivity contains an invalid node number."
        )

    if np.any(connectivity[:, 0] == connectivity[:, 1]):
        raise ValueError(
            "An element cannot connect a node to itself."
        )

    if np.any(k_values <= 0) or np.any(EI_values <= 0):
        raise ValueError(
            "EA/L and EI must be positive for all elements."
        )

    # --------------------------------------------------------
    # Assemble global stiffness and distributed loads
    # --------------------------------------------------------

    K, Fp = assemble_global_system(
        num_nodes,
        coords,
        connectivity,
        k_values,
        EI_values,
        distributed_loads
    )

    # External nodal loads
    F_external = build_force_vector(
        loads_input, num_nodes
    )

    # Total global load vector
    F = F_external + Fp

    # --------------------------------------------------------
    # Solve system
    # --------------------------------------------------------

    u, reactions, free_dofs = solve_system(
        K, F, prescribed_displacements
    )

    # --------------------------------------------------------
    # Recover local element end forces
    # --------------------------------------------------------

    internal_forces = recover_element_forces(
        coords,
        u,
        connectivity,
        k_values,
        EI_values,
        distributed_loads
    )

    # --------------------------------------------------------
    # Verification
    # --------------------------------------------------------

    check_solution(
        K,
        F,
        u,
        reactions,
        free_dofs,
        coords,
        connectivity,
        internal_forces
    )

    # --------------------------------------------------------
    # Write output files
    # --------------------------------------------------------

    nodal_displacements = u.reshape(
        (num_nodes, 3)
    )

    reaction_pairs = reactions.reshape(
        (num_nodes, 3)
    )

    write_list_to_file(
        "nodal displacements.txt",
        nodal_displacements.tolist()
    )

    write_list_to_file(
        "reaction forces.txt",
        reaction_pairs.tolist()
    )

    write_list_to_file(
        "internal forces.txt",
        internal_forces.tolist()
    )

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("\nGlobal stiffness matrix K:")
    print(K)

    print("\nNodal displacements [ux, uy, theta]:")
    print(nodal_displacements.tolist())

    print("\nReaction forces [Rx, Ry, Mz]:")
    print(reaction_pairs.tolist())

    print("\nInternal element forces [N1, V1, M1, N2, V2, M2]:")
    print(internal_forces.tolist())

    print("\nOutput files created successfully.")


if __name__ == "__main__":
    main()

