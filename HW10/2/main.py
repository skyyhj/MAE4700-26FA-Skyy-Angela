import ast
import math
import numpy as np


def read_list_file(filename):
    """Safely parses a text file containing a raw Python list."""
    with open(filename, "r", encoding="utf-8") as f:
        content = f.read().strip()
    return ast.literal_eval(content)


def write_list_file(filename, data):
    """Writes a Python list to a text file in standard format."""
    with open(filename, "w", encoding="utf-8") as f:
        f.write(repr(data) + "\n")


def shape_functions_and_derivs(elem_type, xi, eta):
    """
    Returns shape functions N (1D array) and natural coordinate derivatives
    dN_dxi, dN_deta (1D arrays) at isoparametric coordinates (xi, eta).
    """
    if elem_type == "T3":
        # Isoparametric triangle: (0,0), (1,0), (0,1)
        N = np.array([1.0 - xi - eta, xi, eta], dtype=float)
        dN_dxi = np.array([-1.0, 1.0, 0.0], dtype=float)
        dN_deta = np.array([-1.0, 0.0, 1.0], dtype=float)
    elif elem_type == "Q4":
        # Isoparametric bilinear quad: (-1,-1), (1,-1), (1,1), (-1,1)
        N = 0.25 * np.array(
            [
                (1.0 - xi) * (1.0 - eta),
                (1.0 + xi) * (1.0 - eta),
                (1.0 + xi) * (1.0 + eta),
                (1.0 - xi) * (1.0 + eta),
            ],
            dtype=float,
        )
        dN_dxi = 0.25 * np.array(
            [-(1.0 - eta), (1.0 - eta), (1.0 + eta), -(1.0 + eta)], dtype=float
        )
        dN_deta = 0.25 * np.array(
            [-(1.0 - xi), -(1.0 + xi), (1.0 + xi), (1.0 - xi)], dtype=float
        )
    else:
        raise ValueError(f"Unsupported element type: {elem_type}")
    return N, dN_dxi, dN_deta


def get_quadrature(elem_type):
    """Returns quadrature points [(xi, eta), ...] and weights [w, ...]."""
    if elem_type == "T3":
        # 1-point quadrature for affine T3 elements (area of reference triangle = 0.5)
        pts = [(1.0 / 3.0, 1.0 / 3.0)]
        wts = [0.5]
    elif elem_type == "Q4":
        # 2x2 Gauss-Legendre quadrature for Q4 elements
        g = 1.0 / math.sqrt(3.0)
        pts = [(-g, -g), (g, -g), (g, g), (-g, g)]
        wts = [1.0, 1.0, 1.0, 1.0]
    else:
        raise ValueError(f"Unsupported element type: {elem_type}")
    return pts, wts


def element_matrices(coords, G_alpha, elem_type):
    """
    Computes element conductance matrix k_e and source vector f_e
    using isoparametric mapping and numerical quadrature.
    """
    num_nodes = len(coords)
    k_e = np.zeros((num_nodes, num_nodes), dtype=float)
    f_e = np.zeros(num_nodes, dtype=float)
    pts, wts = get_quadrature(elem_type)

    for (xi, eta), w in zip(pts, wts):
        N, dN_dxi, dN_deta = shape_functions_and_derivs(elem_type, xi, eta)

        # Jacobian matrix J = [dx/dxi, dy/dxi; dx/deta, dy/deta]
        J = np.zeros((2, 2), dtype=float)
        J[0, 0] = np.dot(dN_dxi, coords[:, 0])
        J[0, 1] = np.dot(dN_dxi, coords[:, 1])
        J[1, 0] = np.dot(dN_deta, coords[:, 0])
        J[1, 1] = np.dot(dN_deta, coords[:, 1])

        detJ = np.linalg.det(J)
        if detJ <= 0.0:
            raise ValueError(
                f"Non-positive Jacobian determinant ({detJ:.6e}). Check node ordering."
            )

        invJ = np.linalg.inv(J)
        # B matrix: [dN/dx; dN/dy] = inv(J) @ [dN/dxi; dN/deta]
        dN_nat = np.vstack((dN_dxi, dN_deta))
        B = invJ @ dN_nat

        weight_factor = detJ * w
        k_e += (B.T @ B) * weight_factor
        f_e += N * (2.0 * G_alpha) * weight_factor

    return k_e, f_e


def solve_torsion(coords, conn, essential_bcs, G_alpha_list):
    """Assembles global system, applies Dirichlet BCs, and solves for nodal phi."""
    num_nodes = len(coords)
    nodes_per_elem = len(conn[0])
    if nodes_per_elem == 3:
        elem_type = "T3"
    elif nodes_per_elem == 4:
        elem_type = "Q4"
    else:
        raise ValueError(f"Unsupported element node count: {nodes_per_elem}")

    K = np.zeros((num_nodes, num_nodes), dtype=float)
    F = np.zeros(num_nodes, dtype=float)

    # Global assembly
    for elem_nodes, G_alpha in zip(conn, G_alpha_list):
        elem_coords = coords[elem_nodes]
        k_e, f_e = element_matrices(elem_coords, G_alpha, elem_type)
        for i_local, i_global in enumerate(elem_nodes):
            F[i_global] += f_e[i_local]
            for j_local, j_global in enumerate(elem_nodes):
                K[i_global, j_global] += k_e[i_local, j_local]

    # Partition DOFs for essential boundary conditions
    fixed_dofs = []
    fixed_vals = []
    free_dofs = []

    for idx, bc_val in enumerate(essential_bcs):
        if bc_val is not None:
            fixed_dofs.append(idx)
            fixed_vals.append(float(bc_val))
        else:
            free_dofs.append(idx)

    phi = np.zeros(num_nodes, dtype=float)
    if fixed_dofs:
        phi[fixed_dofs] = np.array(fixed_vals, dtype=float)

    if free_dofs:
        K_ff = K[np.ix_(free_dofs, free_dofs)]
        if fixed_dofs:
            K_fc = K[np.ix_(free_dofs, fixed_dofs)]
            rhs = F[free_dofs] - K_fc @ phi[fixed_dofs]
        else:
            rhs = F[free_dofs]
        phi[free_dofs] = np.linalg.solve(K_ff, rhs)

    return phi, elem_type


def post_process(coords, conn, phi, elem_type):
    """
    Computes element centroidal stresses [tau_xz, tau_yz, sigma_vm]
    and integrates total torque T = 2 * integral(phi dOmega) using element quadrature.
    """
    element_stresses = []
    total_torque = 0.0

    # Centroid in isoparametric space
    if elem_type == "T3":
        xi_c, eta_c = 1.0 / 3.0, 1.0 / 3.0
    else:
        xi_c, eta_c = 0.0, 0.0

    pts, wts = get_quadrature(elem_type)

    for elem_nodes in conn:
        elem_coords = coords[elem_nodes]
        elem_phi = phi[elem_nodes]

        # 1. Centroidal stress evaluation
        _, dN_dxi_c, dN_deta_c = shape_functions_and_derivs(elem_type, xi_c, eta_c)
        J_c = np.array(
            [
                [
                    np.dot(dN_dxi_c, elem_coords[:, 0]),
                    np.dot(dN_dxi_c, elem_coords[:, 1]),
                ],
                [
                    np.dot(dN_deta_c, elem_coords[:, 0]),
                    np.dot(dN_deta_c, elem_coords[:, 1]),
                ],
            ],
            dtype=float,
        )
        B_c = np.linalg.inv(J_c) @ np.vstack((dN_dxi_c, dN_deta_c))
        grad_phi = B_c @ elem_phi  # [dphi/dx, dphi/dy]

        tau_xz = float(grad_phi[1])
        tau_yz = float(-grad_phi[0])
        sigma_vm = float(math.sqrt(3.0 * (tau_xz**2 + tau_yz**2)))
        element_stresses.append([tau_xz, tau_yz, sigma_vm])

        # 2. Torque integration using same quadrature rule as stiffness assembly
        for (xi, eta), w in zip(pts, wts):
            N, dN_dxi, dN_deta = shape_functions_and_derivs(elem_type, xi, eta)
            J = np.array(
                [
                    [
                        np.dot(dN_dxi, elem_coords[:, 0]),
                        np.dot(dN_dxi, elem_coords[:, 1]),
                    ],
                    [
                        np.dot(dN_deta, elem_coords[:, 0]),
                        np.dot(dN_deta, elem_coords[:, 1]),
                    ],
                ],
                dtype=float,
            )
            detJ = np.linalg.det(J)
            phi_qp = np.dot(N, elem_phi)
            total_torque += 2.0 * phi_qp * detJ * w

    return element_stresses, [float(total_torque)]


def main():
    coords_list = read_list_file("nodal_coordinates.txt")
    conn = read_list_file("connectivity_array.txt")
    essential_bcs = read_list_file("essential_BCs.txt")
    G_alpha_list = read_list_file("problem_parameters.txt")

    coords = np.array(coords_list, dtype=float)
    phi, elem_type = solve_torsion(coords, conn, essential_bcs, G_alpha_list)
    element_stresses, torque_result = post_process(coords, conn, phi, elem_type)

    write_list_file("nodal_phi.txt", [float(val) for val in phi])
    write_list_file("element_stresses.txt", element_stresses)
    write_list_file("torsion_results.txt", torque_result)


if __name__ == "__main__":
    main()
