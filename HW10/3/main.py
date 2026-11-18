import ast
import math
import numpy as np


# ------------------------------------------------------------
# Input / output utilities
# ------------------------------------------------------------

def read_list_file(filename):
    """Read a text file containing a raw Python list."""

    with open(filename, "r", encoding="utf-8") as f:
        content = f.read().strip()

    return ast.literal_eval(content)


def write_list_file(filename, data):
    """Write a Python list to a text file."""

    with open(filename, "w", encoding="utf-8") as f:
        f.write(repr(data) + "\n")


# ------------------------------------------------------------
# Shape functions and derivatives
# ------------------------------------------------------------

def shape_functions_and_derivs(elem_type, xi, eta):
    """
    Returns shape functions N and natural coordinate
    derivatives dN_dxi and dN_deta.
    """

    if elem_type == "T3":

        # Isoparametric triangle:
        # (0,0), (1,0), (0,1)

        N = np.array([
            1.0 - xi - eta,
            xi,
            eta
        ], dtype=float)

        dN_dxi = np.array([
            -1.0, 1.0, 0.0
        ], dtype=float)

        dN_deta = np.array([
            -1.0, 0.0, 1.0
        ], dtype=float)

    elif elem_type == "Q4":

        # Isoparametric bilinear quadrilateral:
        # (-1,-1), (1,-1), (1,1), (-1,1)

        N = 0.25 * np.array([
            (1.0 - xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 + eta),
            (1.0 - xi) * (1.0 + eta)
        ], dtype=float)

        dN_dxi = 0.25 * np.array([
            -(1.0 - eta),
            (1.0 - eta),
            (1.0 + eta),
            -(1.0 + eta)
        ], dtype=float)

        dN_deta = 0.25 * np.array([
            -(1.0 - xi),
            -(1.0 + xi),
            (1.0 + xi),
            (1.0 - xi)
        ], dtype=float)

    else:

        raise ValueError(
            f"Unsupported element type: {elem_type}"
        )

    return N, dN_dxi, dN_deta


# ------------------------------------------------------------
# Gauss quadrature
# ------------------------------------------------------------

def get_quadrature(elem_type):
    """Returns quadrature points and weights."""

    if elem_type == "T3":

        # One-point quadrature for a linear triangle

        pts = [
            (1.0 / 3.0, 1.0 / 3.0)
        ]

        wts = [0.5]

    elif elem_type == "Q4":

        # 2x2 Gauss-Legendre quadrature

        g = 1.0 / math.sqrt(3.0)

        pts = [
            (-g, -g),
            (g, -g),
            (g, g),
            (-g, g)
        ]

        wts = [
            1.0, 1.0, 1.0, 1.0
        ]

    else:

        raise ValueError(
            f"Unsupported element type: {elem_type}"
        )

    return pts, wts


# ------------------------------------------------------------
# Element conductance matrix
# ------------------------------------------------------------

def element_matrices(coords, kappa, elem_type):
    """
    Computes the conductive element matrix:

        K_e = integral(kappa * B.T @ B dOmega)
    """

    num_nodes = len(coords)

    k_e = np.zeros(
        (num_nodes, num_nodes),
        dtype=float
    )

    pts, wts = get_quadrature(elem_type)

    for (xi, eta), w in zip(pts, wts):

        N, dN_dxi, dN_deta = shape_functions_and_derivs(
            elem_type, xi, eta
        )

        # Jacobian matrix

        J = np.zeros((2, 2), dtype=float)

        J[0, 0] = np.dot(dN_dxi, coords[:, 0])
        J[0, 1] = np.dot(dN_dxi, coords[:, 1])
        J[1, 0] = np.dot(dN_deta, coords[:, 0])
        J[1, 1] = np.dot(dN_deta, coords[:, 1])

        detJ = np.linalg.det(J)

        if detJ <= 0.0:

            raise ValueError(
                "Non-positive Jacobian determinant. "
                "Check node ordering."
            )

        invJ = np.linalg.inv(J)

        # Transform natural coordinate derivatives
        # into physical coordinate derivatives

        dN_nat = np.vstack(
            (dN_dxi, dN_deta)
        )

        B = invJ @ dN_nat

        weight_factor = detJ * w

        # Conductive element matrix

        k_e += (
            kappa * (B.T @ B)
        ) * weight_factor

    return k_e


# ------------------------------------------------------------
# Convection boundary contributions
# ------------------------------------------------------------

def convection_edge_matrices(coords, h, T_infinity):
    """
    Calculates the exact convection conductance
    matrix and flux vector for a two-node edge.

    K_h = h * L_e / 6 * [[2, 1], [1, 2]]

    F_h = h * T_infinity * L_e / 2 * [1, 1]
    """

    edge_length = np.linalg.norm(
        coords[1] - coords[0]
    )

    if edge_length <= 0.0:

        raise ValueError(
            "Convection edge has zero length."
        )

    # Convection conductance matrix

    k_h = (
        h * edge_length / 6.0
    ) * np.array([
        [2.0, 1.0],
        [1.0, 2.0]
    ], dtype=float)

    # Convection flux vector

    f_h = (
        h * T_infinity * edge_length / 2.0
    ) * np.array([
        1.0,
        1.0
    ], dtype=float)

    return k_h, f_h


# ------------------------------------------------------------
# Input validation
# ------------------------------------------------------------

def validate_inputs(
    coords,
    conn,
    kappa_list,
    essential_bcs,
    convection_bcs,
    point_sources
):
    """Checks input dimensions and node numbering."""

    num_nodes = len(coords)
    num_elements = len(conn)

    if coords.shape != (num_nodes, 2):

        raise ValueError(
            "Each node must contain x and y coordinates."
        )

    if num_elements == 0:

        raise ValueError(
            "Connectivity array contains no elements."
        )

    nodes_per_elem = len(conn[0])

    if nodes_per_elem not in (3, 4):

        raise ValueError(
            "Only T3 and Q4 elements are supported."
        )

    for elem_nodes in conn:

        if len(elem_nodes) != nodes_per_elem:

            raise ValueError(
                "All elements must have the same type."
            )

        for node in elem_nodes:

            if node < 0 or node >= num_nodes:

                raise ValueError(
                    "Connectivity contains an invalid node."
                )

    if len(kappa_list) != num_elements:

        raise ValueError(
            "Each element must have one conductivity value."
        )

    for kappa in kappa_list:

        if kappa <= 0.0:

            raise ValueError(
                "Conductivity must be positive."
            )

    if len(essential_bcs) != num_nodes:

        raise ValueError(
            "Each node must have one essential BC entry."
        )

    if len(point_sources) != num_nodes:

        raise ValueError(
            "Each node must have one point-source entry."
        )

    # Count element edges to identify exterior boundaries

    edge_counts = {}

    for elem_nodes in conn:

        for i in range(len(elem_nodes)):

            node1 = elem_nodes[i]
            node2 = elem_nodes[(i + 1) % len(elem_nodes)]

            edge_key = tuple(sorted([node1, node2]))

            edge_counts[edge_key] = (
                edge_counts.get(edge_key, 0) + 1
            )

    used_edges = set()

    # Check convection boundary conditions

    for edge in convection_bcs:

        if len(edge) != 4:

            raise ValueError(
                "Convection BCs must have the format "
                "[node1, node2, h, T_infinity]."
            )

        node1, node2, h, T_infinity = edge

        if (
            node1 < 0
            or node1 >= num_nodes
            or node2 < 0
            or node2 >= num_nodes
        ):

            raise ValueError(
                "Convection BC contains an invalid node."
            )

        if h < 0.0:

            raise ValueError(
                "Convection coefficient cannot be negative."
            )

        edge_key = tuple(sorted([node1, node2]))

        if edge_counts.get(edge_key, 0) != 1:

            raise ValueError(
                "Convection must be applied to an "
                "exterior element edge."
            )

        if edge_key in used_edges:

            raise ValueError(
                "Duplicate convection boundary edge."
            )

        used_edges.add(edge_key)


# ------------------------------------------------------------
# Global assembly and heat transfer solution
# ------------------------------------------------------------

def solve_heat_transfer(
    coords,
    conn,
    essential_bcs,
    kappa_list,
    convection_bcs,
    point_sources
):
    """
    Assembles the global conductance matrix and
    flux vector, applies boundary conditions,
    and solves for nodal temperatures.
    """

    num_nodes = len(coords)
    nodes_per_elem = len(conn[0])

    if nodes_per_elem == 3:

        elem_type = "T3"

    elif nodes_per_elem == 4:

        elem_type = "Q4"

    else:

        raise ValueError(
            "Unsupported element node count."
        )

    K = np.zeros(
        (num_nodes, num_nodes),
        dtype=float
    )

    F = np.zeros(num_nodes, dtype=float)

    # --------------------------------------------------------
    # Assemble conductive element matrices
    # --------------------------------------------------------

    for elem_nodes, kappa in zip(
        conn, kappa_list
    ):

        elem_coords = coords[elem_nodes]

        k_e = element_matrices(
            elem_coords,
            kappa,
            elem_type
        )

        for i_local, i_global in enumerate(
            elem_nodes
        ):

            for j_local, j_global in enumerate(
                elem_nodes
            ):

                K[i_global, j_global] += \
                    k_e[i_local, j_local]

    # --------------------------------------------------------
    # Assemble convection matrices and flux vectors
    # --------------------------------------------------------

    for edge in convection_bcs:

        node1, node2, h, T_infinity = edge

        edge_nodes = [
            node1, node2
        ]

        edge_coords = coords[edge_nodes]

        k_h, f_h = convection_edge_matrices(
            edge_coords,
            h,
            T_infinity
        )

        for i_local, i_global in enumerate(
            edge_nodes
        ):

            F[i_global] += f_h[i_local]

            for j_local, j_global in enumerate(
                edge_nodes
            ):

                K[i_global, j_global] += \
                    k_h[i_local, j_local]

    # --------------------------------------------------------
    # Add nodal point sources
    # --------------------------------------------------------

    for i in range(num_nodes):

        F[i] += point_sources[i]

    # --------------------------------------------------------
    # Identify fixed and free degrees of freedom
    # --------------------------------------------------------

    fixed_dofs = []
    fixed_vals = []
    free_dofs = []

    for idx, bc_val in enumerate(
        essential_bcs
    ):

        if bc_val is not None:

            fixed_dofs.append(idx)

            fixed_vals.append(float(bc_val))

        else:

            free_dofs.append(idx)

    # A unique steady-state solution requires a prescribed
    # temperature or a positive convection coefficient.

    if not fixed_dofs and not any(
        edge[2] > 0.0 for edge in convection_bcs
    ):

        raise ValueError(
            "No prescribed temperature or positive "
            "convection boundary condition."
        )

    T = np.zeros(num_nodes, dtype=float)

    if fixed_dofs:

        T[fixed_dofs] = np.array(
            fixed_vals,
            dtype=float
        )

    # --------------------------------------------------------
    # Partition and solve the system
    # --------------------------------------------------------

    if free_dofs:

        K_ff = K[np.ix_(
            free_dofs,
            free_dofs
        )]

        if fixed_dofs:

            K_fc = K[np.ix_(
                free_dofs,
                fixed_dofs
            )]

            rhs = (
                F[free_dofs]
                - K_fc @ T[fixed_dofs]
            )

        else:

            rhs = F[free_dofs]

        try:

            T[free_dofs] = np.linalg.solve(
                K_ff,
                rhs
            )

        except np.linalg.LinAlgError:

            raise ValueError(
                "The global conductance matrix is singular. "
                "Check boundary conditions and connectivity."
            )

    # Thermal reaction / residual vector

    reactions = K @ T - F

    return T, elem_type, K, F, reactions, free_dofs


# ------------------------------------------------------------
# Postprocessing
# ------------------------------------------------------------

def post_process(
    coords,
    conn,
    T,
    elem_type,
    kappa_list
):
    """
    Calculates element centroidal heat fluxes:

        q = -kappa * B @ T_e

    Returns a list of [q_x, q_y] pairs.
    """

    element_heat_fluxes = []

    # Element centroid in natural coordinates

    if elem_type == "T3":

        xi_c = 1.0 / 3.0
        eta_c = 1.0 / 3.0

    else:

        xi_c = 0.0
        eta_c = 0.0

    for elem_nodes, kappa in zip(
        conn, kappa_list
    ):

        elem_coords = coords[elem_nodes]

        elem_T = T[elem_nodes]

        _, dN_dxi, dN_deta = shape_functions_and_derivs(
            elem_type,
            xi_c,
            eta_c
        )

        # Jacobian at the element centroid

        J = np.array([
            [
                np.dot(dN_dxi, elem_coords[:, 0]),
                np.dot(dN_dxi, elem_coords[:, 1])
            ],
            [
                np.dot(dN_deta, elem_coords[:, 0]),
                np.dot(dN_deta, elem_coords[:, 1])
            ]
        ], dtype=float)

        detJ = np.linalg.det(J)

        if detJ <= 0.0:

            raise ValueError(
                "Non-positive centroidal Jacobian."
            )

        B = np.linalg.inv(J) @ np.vstack(
            (dN_dxi, dN_deta)
        )

        # Temperature gradient

        grad_T = B @ elem_T

        # Fourier's law

        q_x = float(
            -kappa * grad_T[0]
        )

        q_y = float(
            -kappa * grad_T[1]
        )

        element_heat_fluxes.append([
            q_x,
            q_y
        ])

    return element_heat_fluxes


# ------------------------------------------------------------
# Verification
# ------------------------------------------------------------

def check_solution(
    coords,
    K,
    F,
    T,
    reactions,
    free_dofs,
    convection_bcs,
    point_sources,
    essential_bcs
):
    """
    Checks matrix symmetry, free-node equilibrium,
    and global energy balance.
    """

    tolerance = 1e-8

    # Check 1: symmetry of the global matrix

    symmetry_check = np.allclose(
        K,
        K.T,
        atol=tolerance
    )

    # Check 2: residual at free nodes

    residual_check = np.allclose(
        reactions[free_dofs],
        0.0,
        atol=tolerance
    )

    # Check 3: calculate total point-source heat input

    total_heat_in = float(
        np.sum(point_sources)
    )

    # Calculate total heat leaving through convection

    total_heat_out = 0.0

    for edge in convection_bcs:

        node1, node2, h, T_infinity = edge

        edge_length = np.linalg.norm(
            coords[node2] - coords[node1]
        )
        T_average = (
            T[node1] + T[node2]
        ) / 2.0
        edge_heat_out = (
            h
            * (T_average - T_infinity)
            * edge_length
        )
        total_heat_out += edge_heat_out
    # Heat supplied by prescribed-temperature boundaries
    fixed_dofs = []
    for i, bc_val in enumerate(essential_bcs):
        if bc_val is not None:
            fixed_dofs.append(i)
    prescribed_heat = float(
        np.sum(reactions[fixed_dofs])
    )
    # Overall energy balance
    energy_error = (
        total_heat_in
        + prescribed_heat
        - total_heat_out
    )
    energy_check = np.isclose(
        energy_error,
        0.0,
        atol=tolerance
    )
    print("\nVerification checks:")
    print(
        "Global conductance matrix symmetric:",
        symmetry_check
    )
    print(
        "Free-node residual approximately zero:",
        residual_check
    )
    print(
        "Total point-source heat input:",
        total_heat_in
    )
    print(
        "Heat supplied by prescribed temperatures:",
        prescribed_heat
    )
    print(
        "Total convective heat output:",
        total_heat_out
    )
    print(
        "Energy balance error:",
        energy_error
    )
    print(
        "Global energy equilibrium satisfied:",
        energy_check
    )
    return (
        symmetry_check
        and residual_check
        and energy_check
    )
# ------------------------------------------------------------
# Main program
# ------------------------------------------------------------
def main():
    # --------------------------------------------------------
    # Read the six standard input files
    # --------------------------------------------------------
    coords_list = read_list_file(
        "nodal_coordinates.txt"
    )
    conn = read_list_file(
        "connectivity_array.txt"
    )
    kappa_list = read_list_file(
        "problem_parameters.txt"
    )
    essential_bcs = read_list_file(
        "essential_BCs.txt"
    )
    convection_bcs = read_list_file(
        "convection_BCs.txt"
    )
    point_sources = read_list_file(
        "point_sources.txt"
    )
    coords = np.array(
        coords_list,
        dtype=float
    )
    # --------------------------------------------------------
    # Validate input data
    # --------------------------------------------------------
    validate_inputs(
        coords,
        conn,
        kappa_list,
        essential_bcs,
        convection_bcs,
        point_sources
    )
    # --------------------------------------------------------
    # Assemble and solve
    # --------------------------------------------------------
    T, elem_type, K, F, reactions, free_dofs = \
        solve_heat_transfer(
            coords,
            conn,
            essential_bcs,
            kappa_list,
            convection_bcs,
            point_sources
        )
    # --------------------------------------------------------
    # Calculate element heat fluxes
    # --------------------------------------------------------
    element_heat_fluxes = post_process(
        coords,
        conn,
        T,
        elem_type,
        kappa_list
    )
    # --------------------------------------------------------
    # Verify solution
    # --------------------------------------------------------
    check_solution(
        coords,
        K,
        F,
        T,
        reactions,
        free_dofs,
        convection_bcs,
        point_sources,
        essential_bcs
    )
    # --------------------------------------------------------
    # Write the two required output files
    # --------------------------------------------------------
    write_list_file(
        "nodal_temperatures.txt",
        T.tolist()
    )
    write_list_file(
        "element_heat_fluxes.txt",
        element_heat_fluxes
    )
    # --------------------------------------------------------
    # Display summary of results
    # --------------------------------------------------------
    print("\nAnalysis results:")
    print("Element type:", elem_type)
    print("Number of nodes:", len(coords))
    print("Number of elements:", len(conn))
    print("Minimum temperature:", float(np.min(T)))
    print("Maximum temperature:", float(np.max(T)))
    print("\nOutput files created:")
    print("  nodal_temperatures.txt")
    print("  element_heat_fluxes.txt")

if __name__ == "__main__":
    main()
