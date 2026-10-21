import ast
from pathlib import Path
import numpy as np


# ------------------------------------------------------------
# Input / output utilities
# ------------------------------------------------------------

def read_literal(filename):
    """Read a raw Python literal from a text file."""
    return ast.literal_eval(Path(filename).read_text().strip())


def write_list_to_file(filename, values):
    """Write a Python list to a text file."""
    Path(filename).write_text(repr(values))


# ------------------------------------------------------------
# Shape functions
# ------------------------------------------------------------

def reference_nodes(nnpe):
    """Return reference node locations for linear, quadratic, or cubic elements."""
    nodes = {
        2: [-1.0, 1.0],
        3: [-1.0, 0.0, 1.0],
        4: [-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0]
    }

    if nnpe not in nodes:
        raise ValueError("Nodes per element must be 2, 3, or 4.")

    return np.array(nodes[nnpe], dtype=float)


def shape_functions(xi, nnpe):
    """Evaluate Lagrange shape functions at parent coordinate xi."""
    xi_nodes = reference_nodes(nnpe)
    N = np.ones(nnpe, dtype=float)

    for i in range(nnpe):
        for j in range(nnpe):
            if i != j:
                N[i] *= (xi - xi_nodes[j]) / (
                    xi_nodes[i] - xi_nodes[j]
                )

    return N


def shape_derivatives(xi, nnpe):
    """Evaluate derivatives of the Lagrange shape functions."""
    xi_nodes = reference_nodes(nnpe)
    dN = np.zeros(nnpe, dtype=float)

    for i in range(nnpe):
        for m in range(nnpe):
            if m == i:
                continue

            term = 1.0 / (xi_nodes[i] - xi_nodes[m])

            for j in range(nnpe):
                if j != i and j != m:
                    term *= (
                        (xi - xi_nodes[j])
                        / (xi_nodes[i] - xi_nodes[j])
                    )

            dN[i] += term

    return dN


# ------------------------------------------------------------
# Gauss quadrature and coefficient functions
# ------------------------------------------------------------

def gauss_rule(nq):
    """Return Gauss-Legendre points and weights."""
    if not isinstance(nq, int) or not 1 <= nq <= 5:
        raise ValueError("Number of quadrature points must be 1 through 5.")

    return np.polynomial.legendre.leggauss(nq)


def evaluate_function(expr, x):
    """Evaluate a coefficient/source expression using x and NumPy."""
    value = eval(expr, {"np": np, "x": x, "__builtins__": {}})

    if isinstance(value, np.ndarray):
        return value
    return float(value)


# ------------------------------------------------------------
# Computational mesh
# ------------------------------------------------------------

def build_computational_mesh(x_end, end_connectivity, nnpe):
    """
    Preserve original end nodes and append internal nodes for
    quadratic/cubic elements.

    Returns:
        x_all: all computational node coordinates
        full_connectivity: connectivity including internal nodes
        original_map: original node index -> computational DOF
    """
    x_end = np.asarray(x_end, dtype=float)

    if x_end.ndim != 1 or len(x_end) < 2:
        raise ValueError("Nodal coordinates must be a 1D list with at least 2 nodes.")

    if not np.all(np.isfinite(x_end)):
        raise ValueError("Nodal coordinates must be finite.")

    if nnpe not in (2, 3, 4):
        raise ValueError("Nodes per element must be 2, 3, or 4.")

    x_all = list(x_end)
    original_map = list(range(len(x_end)))
    full_connectivity = []

    for element in end_connectivity:
        if len(element) != 2:
            raise ValueError("Each connectivity entry must contain two end nodes.")

        left, right = map(int, element)

        if not (0 <= left < len(x_end) and 0 <= right < len(x_end)):
            raise ValueError("Connectivity contains an invalid node index.")

        xL = float(x_end[left])
        xR = float(x_end[right])

        if xR <= xL:
            raise ValueError(
                "Each element must connect its left node to its right node."
            )

        nodes = [left]

        for j in range(1, nnpe - 1):
            fraction = j / (nnpe - 1)
            x_internal = xL + fraction * (xR - xL)
            nodes.append(len(x_all))
            x_all.append(x_internal)

        nodes.append(right)
        full_connectivity.append(nodes)

    return (
        np.asarray(x_all, dtype=float),
        full_connectivity,
        original_map
    )


# ------------------------------------------------------------
# Element stiffness and distributed force
# ------------------------------------------------------------

def element_matrices(xe, nnpe, xi_q, w_q, p_expr, f_expr):
    """
    Compute:
        Ke = integral(B.T * p(x) * B dx)
        Fe = integral(N.T * f(x) dx)
    """
    ke = np.zeros((nnpe, nnpe), dtype=float)
    fe = np.zeros(nnpe, dtype=float)

    for xi, weight in zip(xi_q, w_q):
        N = shape_functions(xi, nnpe)
        dN_dxi = shape_derivatives(xi, nnpe)

        J = float(dN_dxi @ xe)

        if J <= 0.0:
            raise ValueError("Element Jacobian must be positive.")

        B = dN_dxi / J
        x = float(N @ xe)

        p = evaluate_function(p_expr, x)
        f = evaluate_function(f_expr, x)

        ke += np.outer(B, B) * p * J * weight
        fe += N * f * J * weight

    return ke, fe


# ------------------------------------------------------------
# Point sources
# ------------------------------------------------------------

def add_point_sources(x_all, full_connectivity, nnpe, point_sources, F):
    """
    Add each point source exactly once using N(x_source).T * source.
    """
    x_min = float(np.min(x_all))
    x_max = float(np.max(x_all))

    for source in point_sources:
        if len(source) != 2:
            raise ValueError("Each point source must be [location, value].")

        xp, magnitude = map(float, source)

        if not x_min <= xp <= x_max:
            raise ValueError("A point source lies outside the domain.")

        owner = None

        # Select a single owner element, including at shared nodes.
        for e, gdofs in enumerate(full_connectivity):
            xe = x_all[np.asarray(gdofs, dtype=int)]
            xL, xR = xe[0], xe[-1]

            if xL <= xp <= xR:
                owner = (e, gdofs, xe)
                break

        if owner is None:
            raise ValueError(f"Could not locate point source at x={xp}.")

        if owner is not None:
            _, gdofs, xe = owner
            xL, xR = xe[0], xe[-1]
            xi = 2.0 * (xp - xL) / (xR - xL) - 1.0
            N = shape_functions(xi, nnpe)
            for a, A in enumerate(gdofs):
                F[A] += N[a] * magnitude


# ------------------------------------------------------------
# Natural boundary conditions
# ------------------------------------------------------------

def apply_natural_bcs(F, x_end, p_expr, boundary_conditions):
    """
    Natural BC means du/dx = prescribed value.

    Left contribution:  -p(x_min) * g_left
    Right contribution: +p(x_max) * g_right
    """
    x_min = float(np.min(x_end))
    x_max = float(np.max(x_end))

    left_index = int(np.argmin(x_end))
    right_index = int(np.argmax(x_end))

    left_type, left_value = boundary_conditions[0]
    right_type, right_value = boundary_conditions[1]

    if left_type == "natural":
        F[left_index] += (
            -evaluate_function(p_expr, x_min) * float(left_value)
        )

    if right_type == "natural":
        F[right_index] += (
            evaluate_function(p_expr, x_max) * float(right_value)
        )

    return left_index, right_index


# ------------------------------------------------------------
# Partitioned solver
# ------------------------------------------------------------

def solve_system(K, F, essential_dofs, prescribed_values):
    """
    Solve K_ff d_f = F_f - K_fe d_e.
    Reactions are calculated from R = Kd - F.
    """
    num_dofs = len(F)

    essential = np.asarray(essential_dofs, dtype=int)
    all_dofs = np.arange(num_dofs, dtype=int)
    free = np.setdiff1d(all_dofs, essential)

    if len(free) == 0:
        raise ValueError("There are no free degrees of freedom.")

    if len(essential) != len(prescribed_values):
        raise ValueError("Essential DOFs and prescribed values do not match.")

    d = np.zeros(num_dofs, dtype=float)
    d[essential] = np.asarray(prescribed_values, dtype=float)

    K_ff = K[np.ix_(free, free)]
    K_fe = K[np.ix_(free, essential)]

    F_f = F[free]
    d_e = d[essential]

    rhs = F_f - K_fe @ d_e

    try:
        d[free] = np.linalg.solve(K_ff, rhs)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "The free stiffness matrix is singular. Check the boundary "
            "conditions and whether the problem is properly constrained."
        ) from exc

    reactions = K @ d - F

    return d, reactions, free


# ------------------------------------------------------------
# Element-end derivatives
# ------------------------------------------------------------

def element_end_derivatives(xe, de, nnpe):
    """Return du/dx at the left and right ends of an element."""
    values = []

    for xi in (-1.0, 1.0):
        dN_dxi = shape_derivatives(xi, nnpe)
        J = float(dN_dxi @ xe)

        if J <= 0.0:
            raise ValueError("Element Jacobian must be positive.")

        B = dN_dxi / J
        values.append(float(B @ de))

    return values


# ------------------------------------------------------------
# Main program
# ------------------------------------------------------------

def main():
    # Read the eight required input files.
    x_end = np.asarray(
        read_literal("nodal coordinates.txt"), dtype=float
    )
    end_connectivity = read_literal("connectivity array.txt")
    nnpe = read_literal("nodes per element.txt")
    nq = read_literal("num quadrature points.txt")
    p_expr = read_literal("p function.txt")
    f_expr = read_literal("f function.txt")
    point_sources = read_literal("point sources.txt")
    boundary_conditions = read_literal("boundary conditions.txt")

    # Validate inputs.
    if not isinstance(nnpe, int) or nnpe not in (2, 3, 4):
        raise ValueError("Nodes per element must be 2, 3, or 4.")

    if not isinstance(nq, int) or not 1 <= nq <= 5:
        raise ValueError("Quadrature points must be an integer from 1 to 5.")

    if len(boundary_conditions) != 2:
        raise ValueError("Provide exactly two boundary conditions.")

    for bc in boundary_conditions:
        if len(bc) != 2 or bc[0] not in ("essential", "natural"):
            raise ValueError(
                'Each boundary condition must be ["essential", value] '
                'or ["natural", value].'
            )

    # Build computational mesh.
    x_all, full_connectivity, original_map = build_computational_mesh(
        x_end, end_connectivity, nnpe
    )

    num_dofs = len(x_all)
    K = np.zeros((num_dofs, num_dofs), dtype=float)
    F = np.zeros(num_dofs, dtype=float)

    xi_q, w_q = gauss_rule(nq)

    # Assemble global stiffness matrix and distributed source vector.
    for gdofs in full_connectivity:
        gdofs_array = np.asarray(gdofs, dtype=int)
        xe = x_all[gdofs_array]

        ke, fe = element_matrices(
            xe, nnpe, xi_q, w_q, p_expr, f_expr
        )

        K[np.ix_(gdofs_array, gdofs_array)] += ke
        F[gdofs_array] += fe

    # Add point sources and natural boundary terms.
    add_point_sources(x_all, full_connectivity, nnpe, point_sources, F)

    left_index, right_index = apply_natural_bcs(
        F, x_end, p_expr, boundary_conditions
    )

    # Collect essential boundary conditions.
    essential_dofs = []
    prescribed_values = []

    for end_index, bc in zip(
        (left_index, right_index), boundary_conditions
    ):
        bc_type, bc_value = bc

        if bc_type == "essential":
            essential_dofs.append(end_index)
            prescribed_values.append(float(bc_value))

    # Solve using the partitioned system.
    d, residual, free_dofs = solve_system(
        K, F, essential_dofs, prescribed_values
    )

    # Required output 1: solution at original input nodes only.
    nodal_solution = [float(d[i]) for i in original_map]

    # Required output 2: reactions at the two domain ends.
    left_type = boundary_conditions[0][0]
    right_type = boundary_conditions[1][0]

    reactions_out = [
        float(residual[left_index]) if left_type == "essential" else 0.0,
        float(residual[right_index]) if right_type == "essential" else 0.0
    ]

    # Required output 3: derivative at each element's left and right ends.
    derivatives_out = []

    for gdofs in full_connectivity:
        gdofs_array = np.asarray(gdofs, dtype=int)
        xe = x_all[gdofs_array]
        de = d[gdofs_array]

        derivatives_out.append(
            element_end_derivatives(xe, de, nnpe)
        )

    # Write the exact three required output files.
    write_list_to_file("nodal solution.txt", nodal_solution)
    write_list_to_file("reactions.txt", reactions_out)
    write_list_to_file("element derivatives.txt", derivatives_out)

    # Verification checks.
    tolerance = 1e-8
    symmetry_check = np.allclose(K, K.T, atol=tolerance)
    free_residual_check = np.allclose(
        residual[free_dofs], 0.0, atol=tolerance
    )

    print("Verification:")
    print("Global stiffness matrix symmetric:", symmetry_check)
    print("Free-DOF residual approximately zero:", free_residual_check)
    print("Maximum absolute free-DOF residual:",
          float(np.max(np.abs(residual[free_dofs]))))

    print("\nNodal solution:", nodal_solution)
    print("Reactions:", reactions_out)
    print("Element derivatives:", derivatives_out)
    print("\nCreated:")
    print("  nodal solution.txt")
    print("  reactions.txt")
    print("  element derivatives.txt")


if __name__ == "__main__":
    main()
