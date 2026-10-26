
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
# Cubic Hermite shape functions
# ------------------------------------------------------------
def hermite_shape_functions(r, L):
    """Return beam shape functions for [w1, theta1, w2, theta2]."""
    return np.array([
        1 - 3*r**2 + 2*r**3,
        L*(r - 2*r**2 + r**3),
        3*r**2 - 2*r**3,
        L*(-r**2 + r**3)
    ], dtype=float)


# ------------------------------------------------------------
# Beam element DOF mapping
# ------------------------------------------------------------
def element_dofs(nodes):
    """Map element nodes [i,j] to [wi,theta_i,wj,theta_j]."""
    i, j = nodes
    return [2*i, 2*i + 1, 2*j, 2*j + 1]


# ------------------------------------------------------------
# Beam element stiffness and consistent load vector
# ------------------------------------------------------------
def beam_element(x1, x2, EI, p):
    L = x2 - x1
    if L <= 0:
        raise ValueError("Beam element must have x2 > x1.")
    if EI <= 0:
        raise ValueError("EI must be positive.")

    ke = (EI / L**3) * np.array([
        [12, 6*L, -12, 6*L],
        [6*L, 4*L**2, -6*L, 2*L**2],
        [-12, -6*L, 12, -6*L],
        [6*L, 2*L**2, -6*L, 4*L**2]
    ], dtype=float)

    fe = (p*L/2) * np.array([1, L/6, 1, -L/6], dtype=float)
    return ke, fe


# ------------------------------------------------------------
# Nodal forces and moments
# ------------------------------------------------------------
def add_applied_loads(F, applied_loads):
    """Add applied nodal forces (dof=0) and moments (dof=1)."""
    for node, dof, value in applied_loads:
        F[2*node + dof] += value


# ------------------------------------------------------------
# Input validation
# ------------------------------------------------------------
def validate_inputs(coords, connectivity, element_EI,
                    distributed_loads, applied_loads, essential_bcs):
    if coords.ndim != 1 or len(coords) < 2 or not np.all(np.isfinite(coords)):
        raise ValueError("Nodal coordinates must be a 1D list of at least two finite values.")

    num_nodes = len(coords)
    num_elements = len(connectivity)
    if num_elements == 0:
        raise ValueError("At least one element is required.")
    if len(element_EI) != num_elements or len(distributed_loads) != num_elements:
        raise ValueError("EI and distributed-load lists must match element count.")

    EI_values = np.asarray(element_EI, dtype=float)
    p_values = np.asarray(distributed_loads, dtype=float)
    if not np.all(np.isfinite(EI_values)) or np.any(EI_values <= 0):
        raise ValueError("Every EI must be finite and positive.")
    if not np.all(np.isfinite(p_values)):
        raise ValueError("Distributed loads must be finite.")

    previous_left = -np.inf
    previous_right_node = None
    for e, nodes in enumerate(connectivity):
        if len(nodes) != 2:
            raise ValueError(f"Element {e} must have exactly two nodes.")
        i, j = nodes
        if type(i) is not int or type(j) is not int:
            raise ValueError("Connectivity node indices must be integers.")
        if not (0 <= i < num_nodes and 0 <= j < num_nodes):
            raise ValueError("Connectivity contains an invalid node index.")
        if coords[j] <= coords[i]:
            raise ValueError("Elements must be oriented left to right.")
        if coords[i] < previous_left:
            raise ValueError("Elements must be listed left to right.")
        if previous_right_node is not None and i != previous_right_node:
            raise ValueError("Consecutive beam elements must share their connecting node.")
        previous_left = coords[i]
        previous_right_node = j

    used_nodes = {node for nodes in connectivity for node in nodes}
    if used_nodes != set(range(num_nodes)):
        raise ValueError("Every listed node must belong to the connected beam mesh.")

    prescribed = set()
    for records, is_bc in ((applied_loads, False), (essential_bcs, True)):
        for record in records:
            if len(record) != 3:
                raise ValueError("Applied loads and BCs must have [node, dof, value].")
            node, dof, value = record
            if type(node) is not int or not (0 <= node < num_nodes):
                raise ValueError("Load/BC node index must be a valid integer.")
            if type(dof) is not int or dof not in (0, 1):
                raise ValueError("DOF must be 0 (w) or 1 (rotation).")
            if not np.isfinite(float(value)):
                raise ValueError("Load/BC values must be finite.")
            if is_bc:
                gdof = 2*node + dof
                if gdof in prescribed:
                    raise ValueError("A DOF cannot be prescribed more than once.")
                prescribed.add(gdof)


# ------------------------------------------------------------
# Partitioned solver (reused from previous homework)
# ------------------------------------------------------------
def solve_system(K, F, essential_dofs, prescribed_values):
    """Solve K_ff d_f = F_f - K_fe d_e and return R = Kd - F."""
    num_dofs = len(F)
    essential = np.asarray(essential_dofs, dtype=int)
    if len(essential) != len(prescribed_values):
        raise ValueError("Essential DOFs and prescribed values do not match.")
    free = np.setdiff1d(np.arange(num_dofs), essential)
    d = np.zeros(num_dofs, dtype=float)
    d[essential] = np.asarray(prescribed_values, dtype=float)

    if len(free) > 0:
        K_ff = K[np.ix_(free, free)]
        K_fe = K[np.ix_(free, essential)]
        rhs = F[free] - K_fe @ d[essential]
        try:
            d[free] = np.linalg.solve(K_ff, rhs)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Free stiffness matrix is singular; check constraints.") from exc

    reactions = K @ d - F
    return d, reactions, free


# ------------------------------------------------------------
# Beam displacement, bending moment, and shear recovery
# ------------------------------------------------------------
def postprocess_element(x1, x2, EI, de):
    L = x2 - x1
    rows_w, rows_M, rows_V = [], [], []
    for r in (0.0, 0.5, 1.0):
        x = x1 + r*L
        N = hermite_shape_functions(r, L)
        N_xx = np.array([
            (-6 + 12*r)/L**2,
            (-4 + 6*r)/L,
            (6 - 12*r)/L**2,
            (-2 + 6*r)/L
        ], dtype=float)
        N_xxx = np.array([12/L**3, 6/L**2, -12/L**3, 6/L**2], dtype=float)

        w = N @ de
        M = EI * (N_xx @ de)
        V = -EI * (N_xxx @ de)
        rows_w.append([float(x), float(w)])
        rows_M.append([float(x), float(M)])
        rows_V.append([float(x), float(V)])
    return rows_w, rows_M, rows_V


# ------------------------------------------------------------
# Main program
# ------------------------------------------------------------
def main():
    # Read the six required input files.
    coords = np.asarray(read_literal("nodal coordinates.txt"), dtype=float)
    connectivity = read_literal("connectivity array.txt")
    element_EI = read_literal("element flexural rigidities.txt")
    distributed_loads = read_literal("distributed loads.txt")
    applied_loads = read_literal("applied loads.txt")
    essential_bcs = read_literal("essential BCs.txt")

    validate_inputs(coords, connectivity, element_EI,
                    distributed_loads, applied_loads, essential_bcs)

    num_nodes = len(coords)
    num_dofs = 2*num_nodes
    K = np.zeros((num_dofs, num_dofs), dtype=float)
    F = np.zeros(num_dofs, dtype=float)

    # Assemble every element's stiffness and consistent distributed load.
    for e, nodes in enumerate(connectivity):
        i, j = nodes
        gdofs = element_dofs(nodes)
        ke, fe = beam_element(coords[i], coords[j],
                              element_EI[e], distributed_loads[e])
        if not np.allclose(ke, ke.T):
            raise ValueError(f"Element {e} stiffness matrix is not symmetric.")
        for a in range(4):
            A = gdofs[a]
            F[A] += fe[a]
            for b in range(4):
                B = gdofs[b]
                K[A, B] += ke[a, b]

    # Add nodal point forces and moments.
    add_applied_loads(F, applied_loads)

    # Prescribed DOFs must stay in input order for reaction output.
    essential_dofs = [2*node + dof for node, dof, value in essential_bcs]
    prescribed_values = [value for node, dof, value in essential_bcs]
    d, residual, free_dofs = solve_system(K, F, essential_dofs, prescribed_values)

    nodal_displacements = d.reshape((num_nodes, 2)).tolist()
    reaction_values = [float(residual[g]) for g in essential_dofs]

    displacement_rows, moment_rows, shear_rows = [], [], []
    for e, nodes in enumerate(connectivity):
        i, j = nodes
        de = d[element_dofs(nodes)]
        rows_w, rows_M, rows_V = postprocess_element(
            coords[i], coords[j], element_EI[e], de
        )
        displacement_rows.extend(rows_w)
        moment_rows.extend(rows_M)
        shear_rows.extend(rows_V)

    # Write the five required files as ordinary Python lists.
    write_list_to_file("nodal displacements.txt", nodal_displacements)
    write_list_to_file("reaction forces.txt", reaction_values)
    write_list_to_file("displacement.txt", displacement_rows)
    write_list_to_file("bending moment.txt", moment_rows)
    write_list_to_file("shear force.txt", shear_rows)

    # Numerical verification.
    symmetry_check = np.allclose(K, K.T)
    free_residual_check = np.allclose(residual[free_dofs], 0.0, atol=1e-8)
    max_residual = (float(np.max(np.abs(residual[free_dofs])))
                    if len(free_dofs) > 0 else 0.0)

    print("Verification:")
    print("Global stiffness matrix symmetric:", symmetry_check)
    print("Free-DOF residual approximately zero:", free_residual_check)
    print("Maximum absolute free-DOF residual:", max_residual)
    print("\nNodal displacements:", nodal_displacements)
    print("Reaction forces:", reaction_values)
    print("Displacement:", displacement_rows)
    print("Bending moment:", moment_rows)
    print("Shear force:", shear_rows)
    print("\nCreated: nodal displacements.txt, reaction forces.txt, displacement.txt,")
    print("bending moment.txt, shear force.txt")


if __name__ == "__main__":
    main()
