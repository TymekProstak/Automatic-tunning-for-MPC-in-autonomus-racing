#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import numpy as np

from casadi import MX, vertcat, sin, cos, atan2, sqrt, tan
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver


# ============================================================
# Constants
# ============================================================

EPS = 1e-8
BIG = 1e9

# ------------------------------------------------------------
# State / input layout
# x = [ey, epsi, vx, vy, r, delta, ddelta, delta_cmd, T]
# u = [u_ddelta_cmd, dT, Mtv]
# ------------------------------------------------------------
NX = 9
NU = 3

IX_EY        = 0
IX_EPSI      = 1
IX_VX        = 2
IX_VY        = 3
IX_R         = 4
IX_DELTA     = 5
IX_DDELTA    = 6
IX_DELTACMD  = 7
IX_T         = 8

IU_DDELTACMD = 0
IU_DT        = 1
IU_MTV       = 2

# ------------------------------------------------------------
# Runtime params:
#
# p = [
#   kappa, v_ref, mux, muy,
#   q_ey, q_epsi, r_u_ddelta_cmd, r_dT, r_Mtv, q_beta_dyn_kin, gamma_progress,
#   q_ey_terminal, q_epsi_terminal
# ]
#
# NOTE:
# - v_ref is kept only to preserve ABI style / debug compatibility.
# - It is NOT used directly in the cost.
# - Bounds are runtime via C++ constraint setters.
# - Slack costs are runtime via zl/zu/Zl/Zu in C++.
# ------------------------------------------------------------
IDX_KAPPA            = 0
IDX_VREF             = 1
IDX_MUX              = 2
IDX_MUY              = 3

IDX_Q_EY             = 4
IDX_Q_EPSI           = 5
IDX_R_U_DDELTACMD    = 6
IDX_R_DT             = 7
IDX_R_MTV            = 8
IDX_Q_BETA_DYN_KIN   = 9
IDX_GAMMA_PROGRESS   = 10

IDX_Q_EY_TERMINAL    = 11
IDX_Q_EPSI_TERMINAL  = 12

NP = 13

# ------------------------------------------------------------
# CONVEX_OVER_NONLINEAR cost dimensions
#
# stage y = [ey, epsi, u_ddelta_cmd, dT, Mtv, beta_dyn_minus_beta_kin, s_dot]
# term  y = [ey, epsi]
# ------------------------------------------------------------
IY_EY                      = 0
IY_EPSI                    = 1
IY_U_DDELTACMD             = 2
IY_D_T                     = 3
IY_MTV                     = 4
IY_BETA_DYN_MINUS_BETA_KIN = 5
IY_S_DOT                   = 6

NY  = 7
NYE = 2


# ============================================================
# Config helpers
# ============================================================

def load_cfg(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)

def cfg_f(d, keys):
    c = d
    for k in keys:
        if isinstance(c, dict) and k in c:
            c = c[k]
        else:
            raise KeyError(f"Missing key: {'/'.join(keys)}")
    return float(c)

def cfg_i(d, keys):
    return int(cfg_f(d, keys))


# ============================================================
# Runtime param defaults
# These are only initial placeholders.
# Final values are meant to be overwritten in runtime from C++.
# ============================================================

def build_default_p(cfg):
    p = np.zeros((NP,), dtype=float)

    p[IDX_KAPPA]           = 0.0
    p[IDX_VREF]            = 0.0
    p[IDX_MUX]             = cfg_f(cfg, ["mpc", "model", "mux"])
    p[IDX_MUY]             = cfg_f(cfg, ["mpc", "model", "muy"])

    p[IDX_Q_EY]            = cfg_f(cfg, ["mpc", "cost", "q_ey"])
    p[IDX_Q_EPSI]          = cfg_f(cfg, ["mpc", "cost", "Q_epsi"])
    p[IDX_R_U_DDELTACMD]   = cfg_f(cfg, ["mpc", "cost", "R_u_ddelta_cmd"])
    p[IDX_R_DT]            = cfg_f(cfg, ["mpc", "cost", "R_dT"])
    p[IDX_R_MTV]           = cfg_f(cfg, ["mpc", "cost", "R_Mtv"])
    p[IDX_Q_BETA_DYN_KIN]  = cfg_f(cfg, ["mpc", "cost", "Q_beta"])
    p[IDX_GAMMA_PROGRESS]  = cfg_f(cfg, ["mpc", "cost", "q_sdot"])

    p[IDX_Q_EY_TERMINAL]   = cfg_f(cfg, ["mpc", "cost", "q_ey_terminal"])
    p[IDX_Q_EPSI_TERMINAL] = cfg_f(cfg, ["mpc", "cost", "Q_epsi_terminal"])

    return p


# ============================================================
# Bounds / slack helpers
# NOTE:
# bounds and slack costs are placeholders only here.
# They are expected to be overwritten in runtime from C++.
# ============================================================

def get_state_lbx_from_cfg(cfg):
    return np.array([
        cfg_f(cfg, ["mpc", "bounds", "min_ey"]),
        cfg_f(cfg, ["mpc", "bounds", "min_epsi"]),
        cfg_f(cfg, ["mpc", "bounds", "min_vx"]),
        cfg_f(cfg, ["mpc", "bounds", "min_vy"]),
        cfg_f(cfg, ["mpc", "bounds", "min_r"]),
        cfg_f(cfg, ["mpc", "bounds", "min_delta"]),
        cfg_f(cfg, ["mpc", "bounds", "min_ddelta_state"]),
        cfg_f(cfg, ["mpc", "bounds", "min_delta_cmd_state"]),
        cfg_f(cfg, ["mpc", "bounds", "min_T"]),
    ], dtype=float)


def get_state_ubx_from_cfg(cfg):
    return np.array([
        cfg_f(cfg, ["mpc", "bounds", "max_ey"]),
        cfg_f(cfg, ["mpc", "bounds", "max_epsi"]),
        cfg_f(cfg, ["mpc", "bounds", "max_vx"]),
        cfg_f(cfg, ["mpc", "bounds", "max_vy"]),
        cfg_f(cfg, ["mpc", "bounds", "max_r"]),
        cfg_f(cfg, ["mpc", "bounds", "max_delta"]),
        cfg_f(cfg, ["mpc", "bounds", "max_ddelta_state"]),
        cfg_f(cfg, ["mpc", "bounds", "max_delta_cmd_state"]),
        cfg_f(cfg, ["mpc", "bounds", "max_T"]),
    ], dtype=float)


def get_input_lbu_from_cfg(cfg):
    return np.array([
        cfg_f(cfg, ["mpc", "bounds", "min_u_ddelta_cmd"]),
        cfg_f(cfg, ["mpc", "bounds", "min_dT"]),
        cfg_f(cfg, ["mpc", "bounds", "min_Mtv"]),
    ], dtype=float)


def get_input_ubu_from_cfg(cfg):
    return np.array([
        cfg_f(cfg, ["mpc", "bounds", "max_u_ddelta_cmd"]),
        cfg_f(cfg, ["mpc", "bounds", "max_dT"]),
        cfg_f(cfg, ["mpc", "bounds", "max_Mtv"]),
    ], dtype=float)


def build_slack_cost_stage(cfg):
    q_slack_track_lin  = cfg_f(cfg, ["mpc", "cost", "q_slack_track_lin"])
    q_slack_track_quad = cfg_f(cfg, ["mpc", "cost", "q_slack_track_quad"])
    q_slack_fric_lin   = cfg_f(cfg, ["mpc", "cost", "q_slack_fric_lin"])
    q_slack_fric_quad  = cfg_f(cfg, ["mpc", "cost", "q_slack_fric_quad"])

    zl = np.array([q_slack_track_lin, q_slack_track_lin, q_slack_fric_lin, q_slack_fric_lin], dtype=float)
    zu = zl.copy()
    Zl = np.array([q_slack_track_quad, q_slack_track_quad, q_slack_fric_quad, q_slack_fric_quad], dtype=float)
    Zu = Zl.copy()

    return zl, zu, Zl, Zu


def build_slack_cost_terminal(cfg):
    return build_slack_cost_stage(cfg)


# ============================================================
# Model
# ============================================================

def create_frenet_model(cfg: dict) -> AcadosModel:
    model = AcadosModel()
    model.name = "frenet_centerline_runtime"

    # --------------------------------------------------------
    # vehicle params
    # --------------------------------------------------------
    m   = cfg_f(cfg, ["mpc", "model", "m"])
    lf  = cfg_f(cfg, ["mpc", "model", "lf"])
    lr  = cfg_f(cfg, ["mpc", "model", "lr"])
    Iz  = cfg_f(cfg, ["mpc", "model", "Iz"])

    B   = cfg_f(cfg, ["mpc", "model", "B"])
    C   = cfg_f(cfg, ["mpc", "model", "C"])
    D   = cfg_f(cfg, ["mpc", "model", "D"])

    Cr0 = cfg_f(cfg, ["mpc", "model", "Cr0"])
    Cd  = cfg_f(cfg, ["mpc", "model", "Cd"])
    Cl  = cfg_f(cfg, ["mpc", "model", "Cl"])
    Cm  = cfg_f(cfg, ["mpc", "model", "Cm"])

    wn   = cfg_f(cfg, ["mpc", "model", "steer_natural_freq"])
    zeta = cfg_f(cfg, ["mpc", "model", "steer_damping"])

    track_width = cfg_f(cfg, ["mpc", "constraints", "track_width"])
    L_c         = cfg_f(cfg, ["mpc", "constraints", "L_c"])
    W_c         = cfg_f(cfg, ["mpc", "constraints", "W_c"])

    g = 9.81
    track_half_width = 0.5 * track_width

    # --------------------------------------------------------
    # states
    # --------------------------------------------------------
    ey        = MX.sym("ey")
    epsi      = MX.sym("epsi")
    vx        = MX.sym("vx")
    vy        = MX.sym("vy")
    r         = MX.sym("r")
    delta     = MX.sym("delta")
    ddelta    = MX.sym("ddelta")
    delta_cmd = MX.sym("delta_cmd")
    T_st      = MX.sym("T")

    x = vertcat(ey, epsi, vx, vy, r, delta, ddelta, delta_cmd, T_st)

    # --------------------------------------------------------
    # inputs
    # --------------------------------------------------------
    u_ddelta_cmd = MX.sym("u_ddelta_cmd")
    dT           = MX.sym("dT")
    Mtv          = MX.sym("Mtv")

    u = vertcat(u_ddelta_cmd, dT, Mtv)

    # --------------------------------------------------------
    # params
    # --------------------------------------------------------
    kappa           = MX.sym("kappa")
    v_ref           = MX.sym("v_ref")
    mux             = MX.sym("mux")
    muy             = MX.sym("muy")

    q_ey            = MX.sym("q_ey")
    q_epsi          = MX.sym("q_epsi")
    r_u_ddelta_cmd  = MX.sym("r_u_ddelta_cmd")
    r_dT            = MX.sym("r_dT")
    r_Mtv           = MX.sym("r_Mtv")
    q_beta_dyn_kin  = MX.sym("q_beta_dyn_kin")
    gamma_progress  = MX.sym("gamma_progress")

    q_ey_terminal   = MX.sym("q_ey_terminal")
    q_epsi_terminal = MX.sym("q_epsi_terminal")

    p = vertcat(
        kappa, v_ref, mux, muy,
        q_ey, q_epsi, r_u_ddelta_cmd, r_dT, r_Mtv, q_beta_dyn_kin, gamma_progress,
        q_ey_terminal, q_epsi_terminal
    )

    # --------------------------------------------------------
    # dynamics
    # --------------------------------------------------------
    vx_safe = sqrt(vx * vx + EPS)

    beta_dyn = atan2(vy, vx_safe)
    beta_kin = atan2(lr * tan(delta), (lf + lr))
    beta_dyn_minus_beta_kin = beta_dyn - beta_kin

    s_dot = (vx * cos(epsi) - vy * sin(epsi)) / (1.0 - ey * kappa)

    FN_net = m * g + Cl * vx * vx
    Nf = FN_net * (lr / (lf + lr))
    Nr = FN_net * (lf / (lf + lr))

    alpha_f = delta - atan2(vy + r * lf, vx_safe)
    alpha_r = -atan2(vy - r * lr, vx_safe)

    Fy_f = Nf * D * sin(C * atan2(B * alpha_f, 1.0))
    Fy_r = Nr * D * sin(C * atan2(B * alpha_r, 1.0))

    F_fric = Cr0 + Cd * vx * vx
    Fx = Cm * T_st

    ey_dot        = vx * sin(epsi) + vy * cos(epsi)
    epsi_dot      = r - kappa * s_dot
    vx_dot        = (1.0 / m) * (Fx * (1.0 + cos(delta)) - Fy_f * sin(delta) + m * vy * r - F_fric)
    vy_dot        = (1.0 / m) * (Fy_r + Fx * sin(delta) + Fy_f * cos(delta) - m * vx * r)
    r_dot         = (1.0 / Iz) * ((Fx * sin(delta) + Fy_f * cos(delta)) * lf - Fy_r * lr + Mtv)
    delta_dot     = ddelta
    ddelta_dot    = -(wn * wn) * delta - 2.0 * zeta * wn * ddelta + (wn * wn) * delta_cmd
    delta_cmd_dot = u_ddelta_cmd
    T_dot         = dT

    xdot = vertcat(
        ey_dot,
        epsi_dot,
        vx_dot,
        vy_dot,
        r_dot,
        delta_dot,
        ddelta_dot,
        delta_cmd_dot,
        T_dot
    )

    model.f_expl_expr = xdot

    # --------------------------------------------------------
    # DISCRETE dynamics (RK2 midpoint)
    # --------------------------------------------------------
    dt = cfg_f(cfg, ["mpc", "solver", "mpc_dt"])

    x_mid = x + 0.5 * dt * xdot

    ey_m        = x_mid[IX_EY]
    epsi_m      = x_mid[IX_EPSI]
    vx_m        = x_mid[IX_VX]
    vy_m        = x_mid[IX_VY]
    r_m         = x_mid[IX_R]
    delta_m     = x_mid[IX_DELTA]
    ddelta_m    = x_mid[IX_DDELTA]
    delta_cmd_m = x_mid[IX_DELTACMD]
    T_m         = x_mid[IX_T]

    vx_safe_m = sqrt(vx_m * vx_m + EPS)
    s_dot_m   = (vx_m * cos(epsi_m) - vy_m * sin(epsi_m)) / (1.0 - ey_m * kappa)

    FN_net_m = m * g + Cl * vx_m * vx_m
    Nf_m = FN_net_m * (lr / (lf + lr))
    Nr_m = FN_net_m * (lf / (lf + lr))

    alpha_f_m = delta_m - atan2(vy_m + r_m * lf, vx_safe_m)
    alpha_r_m = -atan2(vy_m - r_m * lr, vx_safe_m)

    Fy_f_m = Nf_m * D * sin(C * atan2(B * alpha_f_m, 1.0))
    Fy_r_m = Nr_m * D * sin(C * atan2(B * alpha_r_m, 1.0))

    F_fric_m = Cr0 + Cd * vx_m * vx_m
    Fx_m = Cm * T_m

    xdot_mid = vertcat(
        vx_m * sin(epsi_m) + vy_m * cos(epsi_m),
        r_m - kappa * s_dot_m,
        (1.0 / m) * (Fx_m * (1.0 + cos(delta_m)) - Fy_f_m * sin(delta_m) + m * vy_m * r_m - F_fric_m),
        (1.0 / m) * (Fy_r_m + Fx_m * sin(delta_m) + Fy_f_m * cos(delta_m) - m * vx_m * r_m),
        (1.0 / Iz) * ((Fx_m * sin(delta_m) + Fy_f_m * cos(delta_m)) * lf - Fy_r_m * lr + Mtv),
        ddelta_m,
        -(wn * wn) * delta_m - 2.0 * zeta * wn * ddelta_m + (wn * wn) * delta_cmd_m,
        u_ddelta_cmd,
        dT
    )

    model.disc_dyn_expr = x + dt * xdot_mid

    # --------------------------------------------------------
    # CONVEX_OVER_NONLINEAR cost
    #
    # stage y = [ey, epsi, u_ddelta_cmd, dT, Mtv, beta_err, s_dot]
    # term  y = [ey, epsi]
    #
    # outer loss:
    #   0.5*q_ey*ey^2 + 0.5*q_epsi*epsi^2
    # + 0.5*r_u_ddelta_cmd*u_ddelta_cmd^2 + 0.5*r_dT*dT^2 + 0.5*r_Mtv*Mtv^2
    # + 0.5*q_beta_dyn_kin*(beta_err)^2
    # - gamma_progress*s_dot
    #
    # All these weights are runtime via p.
    # --------------------------------------------------------
    stage_y = vertcat(
        ey,
        epsi,
        u_ddelta_cmd,
        dT,
        Mtv,
        beta_dyn_minus_beta_kin,
        s_dot
    )

    term_y = vertcat(
        ey,
        epsi
    )

    model.cost_y_expr   = stage_y
    model.cost_y_expr_0 = stage_y
    model.cost_y_expr_e = term_y

    r_stage   = MX.sym("r_stage", NY)
    r_stage_0 = MX.sym("r_stage_0", NY)
    r_term    = MX.sym("r_term", NYE)

    model.cost_r_in_psi_expr   = r_stage
    model.cost_r_in_psi_expr_0 = r_stage_0
    model.cost_r_in_psi_expr_e = r_term

    psi_stage = (
        0.5 * q_ey             * r_stage[IY_EY] ** 2
        + 0.5 * q_epsi         * r_stage[IY_EPSI] ** 2
        + 0.5 * r_u_ddelta_cmd * r_stage[IY_U_DDELTACMD] ** 2
        + 0.5 * r_dT           * r_stage[IY_D_T] ** 2
        + 0.5 * r_Mtv          * r_stage[IY_MTV] ** 2
        + 0.5 * q_beta_dyn_kin * r_stage[IY_BETA_DYN_MINUS_BETA_KIN] ** 2
        -       gamma_progress * r_stage[IY_S_DOT]
    )

    psi_stage_0 = (
        0.5 * q_ey             * r_stage_0[IY_EY] ** 2
        + 0.5 * q_epsi         * r_stage_0[IY_EPSI] ** 2
        + 0.5 * r_u_ddelta_cmd * r_stage_0[IY_U_DDELTACMD] ** 2
        + 0.5 * r_dT           * r_stage_0[IY_D_T] ** 2
        + 0.5 * r_Mtv          * r_stage_0[IY_MTV] ** 2
        + 0.5 * q_beta_dyn_kin * r_stage_0[IY_BETA_DYN_MINUS_BETA_KIN] ** 2
        -       gamma_progress * r_stage_0[IY_S_DOT]
    )

    psi_term = (
        0.5 * q_ey_terminal   * r_term[0] ** 2
        + 0.5 * q_epsi_terminal * r_term[1] ** 2
    )

    model.cost_psi_expr   = psi_stage
    model.cost_psi_expr_0 = psi_stage_0
    model.cost_psi_expr_e = psi_term

    # --------------------------------------------------------
    # nonlinear constraints = track + friction
    # --------------------------------------------------------
    epsi_abs = sqrt(epsi * epsi + EPS)

    track_left  =  ey + L_c * sin(epsi_abs) + W_c * cos(epsi) - track_half_width
    track_right = -ey + L_c * sin(epsi_abs) + W_c * cos(epsi) - track_half_width

    Nf_safe = sqrt(Nf * Nf + 1.0)
    Nr_safe = sqrt(Nr * Nr + 1.0)

    h_fric_f = (Fx / (mux * Nf_safe))**2 + (Fy_f / (muy * Nf_safe))**2 - 1.0
    h_fric_r = (Fx / (mux * Nr_safe))**2 + (Fy_r / (muy * Nr_safe))**2 - 1.0

    h_stage = vertcat(track_left, track_right, h_fric_f, h_fric_r)
    h_term  = vertcat(track_left, track_right, h_fric_f, h_fric_r)

    model.con_h_expr   = h_stage
    model.con_h_expr_0 = h_stage
    model.con_h_expr_e = h_term

    model.x = x
    model.u = u
    model.p = p
    model.z = vertcat([])

    return model


# ============================================================
# OCP
# ============================================================

def create_ocp(cfg, codegen_dir, ocp_json):
    N_h = cfg_i(cfg, ["mpc", "solver", "mpc_N"])
    dt  = cfg_f(cfg, ["mpc", "solver", "mpc_dt"])

    model = create_frenet_model(cfg)
    nx = model.x.size1()
    nu = model.u.size1()

    if nx != NX or nu != NU:
        raise RuntimeError(f"Unexpected dims: nx={nx}, nu={nu}, expected NX={NX}, NU={NU}")

    ocp = AcadosOcp()
    ocp.model = model

    ocp.solver_options.N_horizon = N_h
    try:
        ocp.solver_options.N = N_h
    except Exception:
        pass

    try:
        ocp.code_gen_opts.code_export_directory = codegen_dir
    except Exception:
        ocp.code_export_directory = codegen_dir

    ocp.solver_options.tf = N_h * dt

    # --------------------------------------------------------
    # Cost
    # --------------------------------------------------------
    ocp.cost.cost_type   = "CONVEX_OVER_NONLINEAR"
    ocp.cost.cost_type_0 = "CONVEX_OVER_NONLINEAR"
    ocp.cost.cost_type_e = "CONVEX_OVER_NONLINEAR"

    ocp.cost.yref   = np.zeros((NY,), dtype=float)
    ocp.cost.yref_0 = np.zeros((NY,), dtype=float)
    ocp.cost.yref_e = np.zeros((NYE,), dtype=float)

    # --------------------------------------------------------
    # Native box bounds
    # PLACEHOLDERS only; runtime C++ overwrites them.
    # --------------------------------------------------------
    ocp.constraints.idxbu = np.arange(NU, dtype=int)
    ocp.constraints.lbu   = get_input_lbu_from_cfg(cfg)
    ocp.constraints.ubu   = get_input_ubu_from_cfg(cfg)

    ocp.constraints.idxbx = np.arange(NX, dtype=int)
    ocp.constraints.lbx   = get_state_lbx_from_cfg(cfg)
    ocp.constraints.ubx   = get_state_ubx_from_cfg(cfg)

    ocp.constraints.idxbx_e = np.arange(NX, dtype=int)
    ocp.constraints.lbx_e   = get_state_lbx_from_cfg(cfg)
    ocp.constraints.ubx_e   = get_state_ubx_from_cfg(cfg)

    ocp.constraints.x0 = np.zeros((NX,), dtype=float)

    # --------------------------------------------------------
    # Nonlinear constraints
    # --------------------------------------------------------
    nh = 4

    ocp.constraints.lh   = -BIG * np.ones((nh,), dtype=float)
    ocp.constraints.uh   = np.zeros((nh,), dtype=float)

    ocp.constraints.lh_0 = -BIG * np.ones((nh,), dtype=float)
    ocp.constraints.uh_0 = np.zeros((nh,), dtype=float)

    ocp.constraints.lh_e = -BIG * np.ones((nh,), dtype=float)
    ocp.constraints.uh_e = np.zeros((nh,), dtype=float)

    # --------------------------------------------------------
    # Soft nonlinear constraints
    # --------------------------------------------------------
    ocp.constraints.idxsh   = np.array([0, 1, 2, 3], dtype=int)
    ocp.constraints.lsh     = np.zeros((4,), dtype=float)
    ocp.constraints.ush     = np.zeros((4,), dtype=float)

    ocp.constraints.idxsh_0 = np.array([0, 1, 2, 3], dtype=int)
    ocp.constraints.lsh_0   = np.zeros((4,), dtype=float)
    ocp.constraints.ush_0   = np.zeros((4,), dtype=float)

    ocp.constraints.idxsh_e = np.array([0, 1, 2, 3], dtype=int)
    ocp.constraints.lsh_e   = np.zeros((4,), dtype=float)
    ocp.constraints.ush_e   = np.zeros((4,), dtype=float)

    zl_stage, zu_stage, Zl_stage, Zu_stage = build_slack_cost_stage(cfg)
    zl_term,  zu_term,  Zl_term,  Zu_term  = build_slack_cost_terminal(cfg)

    ocp.cost.zl   = zl_stage
    ocp.cost.zu   = zu_stage
    ocp.cost.Zl   = Zl_stage
    ocp.cost.Zu   = Zu_stage

    ocp.cost.zl_0 = zl_stage.copy()
    ocp.cost.zu_0 = zu_stage.copy()
    ocp.cost.Zl_0 = Zl_stage.copy()
    ocp.cost.Zu_0 = Zu_stage.copy()

    ocp.cost.zl_e = zl_term
    ocp.cost.zu_e = zu_term
    ocp.cost.Zl_e = Zl_term
    ocp.cost.Zu_e = Zu_term

    # --------------------------------------------------------
    # default p
    # --------------------------------------------------------
    ocp.parameter_values = build_default_p(cfg)

    # --------------------------------------------------------
    # Solver options
    # --------------------------------------------------------
    ocp.solver_options.integrator_type = "DISCRETE"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.nlp_solver_max_iter = 5
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"

    ocp.solver_options.globalization = "MERIT_BACKTRACKING"
    ocp.solver_options.globalization_line_search_use_sufficient_descent = 1
    ocp.solver_options.globalization_use_SOC = 1

    ocp.solver_options.regularize_method = "PROJECT"
    ocp.solver_options.reg_adaptive_eps = True
    ocp.solver_options.hpipm_mode = "ROBUST"

    return ocp


# ============================================================
# Build
# ============================================================

def build_solver(cfg, codegen_dir, ocp_json, force):
    ocp = create_ocp(cfg, codegen_dir, ocp_json)

    if not force:
        try:
            return AcadosOcpSolver(ocp, json_file=ocp_json, build=False, generate=False)
        except TypeError:
            return AcadosOcpSolver(ocp, json_file=ocp_json)

    try:
        return AcadosOcpSolver(ocp, json_file=ocp_json, build=True, generate=True)
    except TypeError:
        return AcadosOcpSolver(ocp, json_file=ocp_json)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--codegen_dir", default="./acados_solver_generated_frenet_centerline_runtime")
    ap.add_argument("--ocp_json", default="acados_ocp_frenet_centerline_runtime.json")
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args()

    cfg = load_cfg(args.json)
    _ = build_solver(cfg, args.codegen_dir, args.ocp_json, args.build)

    print("\n[OK] Frenet solver generated.")
    print("  x = [ey, epsi, vx, vy, r, delta, ddelta, delta_cmd, T]")
    print("  u = [u_ddelta_cmd, dT, Mtv]")
    print("  p = [kappa, v_ref, mux, muy, q_ey, q_epsi, r_u_ddelta_cmd, r_dT, r_Mtv, q_beta_dyn_kin, gamma_progress, q_ey_terminal, q_epsi_terminal]")
    print("  cost = CONVEX_OVER_NONLINEAR")
    print("  stage y = [ey, epsi, u_ddelta_cmd, dT, Mtv, beta_dyn - beta_kin, s_dot]")
    print("  term  y = [ey, epsi]")
    print("  runtime via p: q_ey, q_epsi, R_u_ddelta_cmd, R_dT, R_Mtv, Q_beta_dyn_kin, gamma_progress, q_ey_terminal, Q_epsi_terminal")
    print("  runtime via C++ setters: bounds + slack costs")
    print("  nonlinear h = [track_left, track_right, fric_f, fric_r]")
    print(f"  NX = {NX}, NU = {NU}, NY = {NY}, NYE = {NYE}, NP = {NP}")


if __name__ == "__main__":
    main()
