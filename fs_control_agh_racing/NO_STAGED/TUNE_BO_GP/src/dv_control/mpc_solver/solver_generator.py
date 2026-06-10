import json
import os
import numpy as np

from casadi import SX, reshape, sin, cos, fabs, vertcat
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver


# ============================================================
#  Ścieżki względem położenia tego pliku (mpc_solver/)
# ============================================================
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(THIS_DIR)  # żeby wszystkie "./..." były względem mpc_solver/

JSON_PATH = os.path.abspath(os.path.join(THIS_DIR, "..", "config", "Params", "control_param.json"))
CODEGEN_DIR = "./acados_solver_generated"
OCP_JSON_FILE = "acados_ocp_ltv.json"


# ============================================================
#  Wczytanie parametrów
# ============================================================
print("Wczytuję JSON z:", JSON_PATH)
with open(JSON_PATH, "r") as f:
    cfg = json.load(f)

# ---- parametry ----
N = int(cfg["mpc"]["solver"]["mpc_N"])
eps = cfg["mpc"]["solver"]["eps"]            # zostawiam (możesz użyć w opcjach)
maxit = cfg["mpc"]["solver"]["max_iter"]     # zostawiam (możesz użyć w opcjach)

delta_max = float(cfg["mpc"]["bounds"]["max_delta"])
delta_min = float(cfg["mpc"]["bounds"]["min_delta"])
ddelta_max = float(cfg["mpc"]["bounds"]["max_ddelta"])
ddelta_min = float(cfg["mpc"]["bounds"]["min_ddelta"])

torque_vectoring_max = float(cfg["mpc"]["bounds"]["max_tv"])
torque_vectoring_min = float(cfg["mpc"]["bounds"]["min_tv"])

dt = 1.0 / float(cfg["general"]["odom_frequency"])

Q_y = float(cfg["mpc"]["cost"]["Q_y"])
Q_psi = float(cfg["mpc"]["cost"]["Q_psi"])
R_delta = float(cfg["mpc"]["cost"]["Q_delta"])
R_d_delta = float(cfg["mpc"]["cost"]["R_ddelta"])

# ====== stałe do track-constraintów z JSON ======
Lc_const = float(cfg["mpc"]["model"]["length"])
Wc_const = float(cfg["mpc"]["model"]["width"])

track_width = float(cfg["mpc"]["constraints"]["track_width"])
track_safety = float(cfg["mpc"]["constraints"]["safety_factor"])
NL_const = 0.5 * track_width * track_safety
NR_const = 0.5 * track_width * track_safety

# ====== wagi slacków (Twoje wymaganie) ======
SLACK_W_LIN = 100.0
SLACK_W_SQ  = 1000.0

NX = 7
NU = 2  # d_delta_request, torque_vectoring

print("Używam parametrów MPC:")
print(f"N = {N}")
print(f"dt = {dt}")
print(f"max_ddelta = {ddelta_max}")
print(f"Lc_const = {Lc_const}")
print(f"Wc_const = {Wc_const}")
print(f"NL_const = {NL_const}")
print(f"NR_const = {NR_const}")
print(f"slack: lin={SLACK_W_LIN}, sq={SLACK_W_SQ}")


# ============================================================
#  Model LTV dyskretny: x_{k+1} = Ad x + Bd u + Kd
#  + track constraints z obrazka jako h(x) <= 0
# ============================================================
def create_ltv_discrete_model():
    model = AcadosModel()

    x = SX.sym("x", NX)
    u = SX.sym("u", NU)

    # ---- p = [Ad_flat, Bd_flat, Kd] ----
    np_Ad = NX * NX
    np_Bd = NX * NU
    np_Kd = NX
    NP = np_Ad + np_Bd + np_Kd

    p = SX.sym("p", NP)

    Ad_flat = p[0:np_Ad]
    Bd_flat = p[np_Ad: np_Ad + np_Bd]
    Kd = p[np_Ad + np_Bd: np_Ad + np_Bd + np_Kd]

    Ad = reshape(Ad_flat, NX, NX)
    Bd = reshape(Bd_flat, NX, NU)

    x_next = Ad @ x + Bd @ u + Kd

    model.x = x
    model.u = u
    model.p = p
    model.disc_dyn_expr = x_next
    model.name = "mpc_ltv_discrete"

    # =========================================================
    # Track constraints (obrazek):
    # n - Lc/2 * sin(|mu|) + Wc/2 * cos(mu) <= NL
    # -n + Lc/2 * sin(|mu|) + Wc/2 * cos(mu) <= NR
    #
    # U Ciebie: mu = epsi, biorę epsi = x[1]
    #           n  = x[0]
    # =========================================================
    n = x[0]
    epsi = x[1]

    Lc = SX(Lc_const)
    Wc = SX(Wc_const)
    NL = SX(NL_const)
    NR = SX(NR_const)

    c1 = n + (Lc / 2.0) * sin(fabs(epsi)) + (Wc / 2.0) * cos(epsi) - NL
    c2 = -n + (Lc / 2.0) * sin(fabs(epsi)) + (Wc / 2.0) * cos(epsi) - NR

    model.con_h_expr = vertcat(c1, c2)

    return model, NP


# ============================================================
#  OCP + solver
# ============================================================
def create_ocp_solver():
    # ACADOS_SOURCE_DIR ustawia mpc_solver/run_solver_gen.sh na External/acados_sysroot
    if "ACADOS_SOURCE_DIR" not in os.environ:
        raise RuntimeError(
            "Brak ACADOS_SOURCE_DIR. Odpal przez mpc_solver/run_solver_gen.sh "
            "(ustawia ACADOS_SOURCE_DIR/LD_LIBRARY_PATH/PATH)."
        )

    acados_src = os.environ["ACADOS_SOURCE_DIR"]

    # sanity-check sysroota
    expected_hdr = os.path.join(acados_src, "include", "acados_c", "ocp_nlp_interface.h")
    expected_so = os.path.join(acados_src, "lib", "libacados.so")
    expected_ll = os.path.join(acados_src, "lib", "link_libs.json")

    if not os.path.isfile(expected_hdr):
        raise RuntimeError(f"ACADOS sysroot nie zawiera headera: {expected_hdr}")
    if not os.path.isfile(expected_so):
        raise RuntimeError(f"ACADOS sysroot nie zawiera biblioteki: {expected_so}")
    if not os.path.isfile(expected_ll):
        raise RuntimeError(f"ACADOS sysroot nie zawiera: {expected_ll} (wymagane przez acados_template)")

    model, NP = create_ltv_discrete_model()

    ocp = AcadosOcp()
    ocp.model = model

    # ---- DIMS ----
    ocp.dims.N = N
    ocp.dims.nx = NX
    ocp.dims.nu = NU
    ocp.dims.np = NP

    # =========================================================
    # HARD: h(x) <= 0  (2 szt.)
    # SOFT: dodajemy slacks -> feasible zawsze
    # =========================================================
    ocp.dims.nh = 2
    ocp.constraints.lh = -1e12 * np.ones((2,))  # -inf
    ocp.constraints.uh = np.zeros((2,))        # <= 0

    # ---- SLACKS na constraints h (2 szt.) ----
    ns = 2
    ocp.dims.ns = ns
    ocp.constraints.idxsh = np.array([0, 1], dtype=int)
    ocp.constraints.lsh = np.zeros((ns,))      # s >= 0
    ocp.constraints.ush = np.zeros((ns,))      # s >= 0

    # kara: 10 * s + 100 * s^2
    ocp.cost.zl = SLACK_W_LIN * np.ones((ns,))
    ocp.cost.zu = SLACK_W_LIN * np.ones((ns,))
    ocp.cost.Zl = SLACK_W_SQ  * np.ones((ns,))
    ocp.cost.Zu = SLACK_W_SQ  * np.ones((ns,))

    # ---- COST (LINEAR_LS) ----
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    ny = NX + NU
    ny_e = NX
    ocp.dims.ny = ny
    ocp.dims.ny_e = ny_e

    ocp.cost.Vx = np.zeros((ny, NX))
    ocp.cost.Vx[:NX, :] = np.eye(NX)

    ocp.cost.Vu = np.zeros((ny, NU))
    ocp.cost.Vu[NX:, :] = np.eye(NU)

    ocp.cost.Vx_e = np.eye(NX)

    # Use an identity matrix (or small positive weights) for compilation
    W = np.eye(ny)
    ocp.cost.W = W
    ocp.cost.yref = np.zeros((ny,))

    W_e = np.eye(ny_e)
    ocp.cost.W_e = W_e
    ocp.cost.yref_e = np.zeros((ny_e,))

    # ---- CONSTRAINTS ----
    ocp.constraints.x0 = np.zeros((NX,))

    # stan delta (x[4])
    ocp.constraints.idxbx = np.array([4], dtype=int)
    ocp.constraints.lbx = np.array([delta_min])
    ocp.constraints.ubx = np.array([delta_max])

    # wejścia: constraints na oba u[0] i u[1]
    ocp.constraints.idxbu = np.array([0, 1], dtype=int)
    ocp.constraints.lbu = np.array([ddelta_min, torque_vectoring_min], dtype=float)
    ocp.constraints.ubu = np.array([ddelta_max, torque_vectoring_max], dtype=float)

    # ---- OPTIONS ----
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "DISCRETE"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.tf = N * dt

    # ---- CODEGEN DIR ----
    try:
        ocp.code_gen_opts.code_export_directory = CODEGEN_DIR
    except Exception:
        ocp.code_export_directory = CODEGEN_DIR

    ocp.parameter_values = np.zeros(NP)

    solver = AcadosOcpSolver(ocp, json_file=OCP_JSON_FILE)
    return solver, NP


if __name__ == "__main__":
    solver, NP = create_ocp_solver()
    print("Solver wygenerowany pomyślnie.")
    print("Track constraints są SOFT (slack), kara: 10 * s + 100 * s^2")
    print("Możesz teraz dynamicznie zmieniać wagi W używając solver.cost_set()")
