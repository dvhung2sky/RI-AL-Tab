"""Ex14-FE: OpenSeesPy model of a STEEL beam-column assembly under a column-removal scenario.

Calibration target: the two full-scale specimens of NIST TN 1669 (Sadek, Lew, Main, Robert,
Chiarito 2010), each three columns + two 20 ft spans, centre stub column pushed down monotonically
under displacement control until the load-carrying capacity was lost.

WHAT THE MODEL MUST REPRODUCE (published values; never used as model input)
    WUF-B (SDC C frame, W21x73 beams, W18x119 columns, welded flanges + bolted web)
        elastic up to a centre-column displacement of about  50 mm  (2 in)
        peak vertical load     890 kN (200 kip) at  495 mm (19.5 in), chord rotation 0.081 rad
        beam axial tension at failure  "slightly greater than"  667 kN (150 kip)
        failure sequence: top-flange local buckling near the centre column -> lowest web bolt ->
                          middle web bolt -> BOTTOM FLANGE fracture at the weld access hole
    RBS   (SDC D frame, W24x94 beams, W24x131 columns, radius-cut reduced beam section)
        elastic up to a centre-column displacement of about  50 mm  (2 in)
        peak vertical load    1780 kN (400 kip) at  851 mm (33.5 in), chord rotation 0.140 rad
        beam axial tension at failure  about  2447 kN (550 kip)
        failure: BOTTOM FLANGE fracture at the middle of the reduced section near the centre column

MECHANISM THE MODEL HAS TO CONTAIN
    The centre column is pushed down, so each span is a fixed-fixed beam with one end displaced:
    DOUBLE curvature.  At the centre-column face the BOTTOM flange is in tension (hence the top
    flange buckles and the bottom flange fractures, exactly as the test reports); at the end-column
    face the top flange is in tension.  As the displacement grows the beam elongates against the
    end-column restraint and axial tension develops -- catenary action -- which is geometric and
    cannot appear without a Corotational transformation.  The connection fractures when the extreme
    tension fibre reaches its fracture strain under COMBINED bending and tension.

WHY THIS MODEL CLASS
    It is the same class the paper's frame evaluator uses (fibre beam-column elements + a short
    weak zone at the fracture location + Corotational + displacement-controlled pushdown), and the
    same class NIST validated in TN 1669 Sections 5.3.2/5.4.2 ("reduced models" of beam and spring
    elements).  Calibrating it here is what licenses its use on the 8-storey frame, where no test
    exists.

UNITS: N, mm, MPa.

INPUT PROVENANCE
    [READ]   read from NIST TN 1669 (Figs 3-1, 3-2, Table 3-1, Secs 2.2.2, 4.2, 4.3, 5.2-5.4)
    [ASSUME] engineering assumption, listed in CALIBRATION.md
"""
import numpy as np
import openseespy.opensees as ops

IN, KIP = 25.4, 4448.2216

# ---- AISC section properties, mm and mm^n -------------------------------------------------
# d = overall depth, bf = flange width, tf = flange thickness, tw = web thickness
W = {
    "W21x73":  dict(d=538.5, bf=210.8, tf=18.8, tw=11.6, A=13871.0, I=6.660e8, Z=2.819e6),
    "W18x119": dict(d=482.6, bf=287.0, tf=26.9, tw=16.6, A=22645.0, I=9.116e8, Z=3.851e6),
    "W24x94":  dict(d=617.2, bf=230.4, tf=22.2, tw=13.1, A=17871.0, I=1.1238e9, Z=4.162e6),
    "W24x131": dict(d=622.3, bf=327.7, tf=34.0, tw=15.4, A=24839.0, I=1.6733e9, Z=6.096e6),
}

ES = 200000.0        # [ASSUME] Es not tabulated in TN 1669; nominal value for structural steel

# ------------------------------------------------------------------------------------------
# Specimens
# ------------------------------------------------------------------------------------------
WUFB = dict(
    name="WUF-B",
    beam="W21x73", col="W18x119",                # [READ] Sec. 2.2, Fig. 3-1
    L=240.0 * IN,                                # [READ] 20 ft centre-to-centre of columns
    h_below=(7.0 * 12 + 1.25) * IN,              # [READ] Fig. 3-1 dimension chain, end column
    h_col=12.0 * 12 * IN,                        # [READ] end column overall height, 12 ft
    # beam flange coupon, Table 3-1.  NIST's own reduced models use FLANGE data for the whole
    # wide-flange section, so the same is done here.
    fy=359.0, fu=496.0, esu=0.174,
    fy_col=352.0,
    rbs=False,
    # weak zone at the centre-column face.  In the test the web bolts shear before the bottom
    # flange fractures; a fibre beam element cannot carry bolt shear, so the bolts' contribution
    # is folded into the calibrated fracture strain of this zone (declared in CALIBRATION.md).
    L_conn=0.5 * 538.5,          # [ASSUME] fracture gauge length, 0.5 db
    theta_u=0.0642,        # [CAL] hinge rotation at fracture, rad -- see CALIBRATION.md
    K_br=35.0e3,                                 # N/mm, horizontal restraint at the end-column top
    pin_base=True,
    u_max=1100.0, du=1.0,
)

RBS = dict(
    name="RBS",
    beam="W24x94", col="W24x131",                # [READ] Sec. 2.2, Fig. 3-2
    L=240.0 * IN,
    h_below=(6.0 * 12 + 5.75) * IN,              # [READ] Fig. 3-2 dimension chain
    h_col=12.0 * 12 * IN,
    # [ASSUME] Table 3-1 gives NO beam-flange coupon for the RBS specimen (web only: 455/554/
    # 0.143).  Scaled by the WUF-B flange/web ratios -- declared, and swept in CALIBRATION.md.
    fy=455.0 * (359.0 / 395.0), fu=554.0 * (496.0 / 500.0), esu=0.143 * (0.174 / 0.189),
    fy_col=378.0,
    rbs=True,
    t_doubler=9.0 / 16.0 * 25.4,   # [READ] Fig. 3-2, centre column only
    # [READ] Sec. 2.2.2: a = 0.55 bf from the column face, b = 0.7 db long, c = 0.25 bf cut per side
    a_fac=0.55, b_fac=0.70, c_fac=0.25,
    L_conn=0.5 * 617.2,                          # [ASSUME] fracture gauge length, 0.5 db
    theta_u=0.1615,        # [CAL] hinge rotation at fracture, rad -- see CALIBRATION.md
    K_br=35.0e3,
    pin_base=True,
    u_max=1000.0, du=1.0,
)

TEST = {
    # rot_incl = the measured inclinometer rotation at failure.  WUF-B: 0.088 rad, stated in
    # Sec. 4.2.3.  RBS: Sec. 4.3.3 gives no number and says the face gauges under-read; the
    # Fig. 4-21/4-22 axes cap the measured value at 3 deg = 0.0524 rad, so it is an ENVELOPE, and
    # the model is required to fall below it, not to match it.
    "WUF-B": dict(u_el=50.0, P_pk=890.0e3, u_pk=495.0, theta=0.081, N_beam=667.0e3,
                  rot_incl=0.088),
    "RBS":   dict(u_el=50.0, P_pk=1780.0e3, u_pk=851.0, theta=0.140, N_beam=2447.0e3,
                  rot_incl=None),
}

# ------------------------------------------------------------------------------------------
# model
# ------------------------------------------------------------------------------------------
NCEN, NCF, NW1, NW2, NEF, NEC, NBASE, NTOP, NGND = 1, 2, 3, 4, 5, 6, 7, 8, 9
NPZC, NPZE = 51, 52          # column-centreline side of the two panel-zone springs
MAT_B, MAT_BF, SEC_FULL, SEC_RBS, SEC_RBS_OUT = 1, 2, 1, 2, 4
EWEAK = 201                      # the element that fractures (centre-column face, or the RBS cut)


def _wsection(tag, s, mat, bf=None):
    """Fibre discretisation of a wide-flange section bending about the strong axis.

    bf overrides the flange width, which is how the radius-cut reduced beam section is
    represented (minimum width at the centre of the cut, where the test fractured).
    """
    bf = s["bf"] if bf is None else bf
    d, tf, tw = s["d"], s["tf"], s["tw"]
    yf, yw = d / 2.0 - tf, d / 2.0 - tf
    ops.section("Fiber", tag)
    ops.patch("rect", mat, 2, 8, yf, -bf / 2.0, d / 2.0, bf / 2.0)      # top flange
    ops.patch("rect", mat, 2, 8, -d / 2.0, -bf / 2.0, -yf, bf / 2.0)    # bottom flange
    ops.patch("rect", mat, 12, 2, -yw, -tw / 2.0, yw, tw / 2.0)         # web
    return tag


def _panel_spring(tag, n_rig, n_beam, sc, sb, t_pz, fy_col):
    """Krawinkler-type panel-zone shear spring, TN 1669 Eqs (5.3)-(5.5) (after Khandelwal 2008).

    The diagonal-spring stiffness k_pz = G(d_c-t_cf)t_pz / [(d_b-t_bf)cos^2(theta)] with
    cos^2(theta) = (d_c-t_cf)^2 / [(d_c-t_cf)^2 + (d_b-t_bf)^2] becomes, as a joint rotational
    stiffness, K_theta = k_pz (d_b-t_bf)^2 cos^2(theta) = G (d_c-t_cf) t_pz (d_b-t_bf) -- every
    term read from the section tables, so there is no free parameter here.

    Without it the joint is rigid and the beam rotation AT THE COLUMN FACE is identically zero,
    which the model cannot then compare against the measured inclinometer readings (0.088 rad at
    failure for the WUF-B; a face rotation far below the chord rotation for the RBS).
    """
    G = ES / 2.6
    dc, tcf, bcf = sc["d"], sc["tf"], sc["bf"]
    db, tbf = sb["d"], sb["tf"]
    K_th = G * (dc - tcf) * t_pz * (db - tbf)
    V_y = 0.6 * fy_col * dc * t_pz * (1.0 + 3.0 * bcf * tcf ** 2 / (db * dc * t_pz))
    M_y = V_y * (db - tbf)
    ops.uniaxialMaterial("Steel01", tag, M_y, K_th, 0.03)
    ops.element("zeroLength", tag, n_rig, n_beam, "-mat", tag, "-dir", 3)
    ops.equalDOF(n_rig, n_beam, 1, 2)
    return K_th, M_y


def build(p):
    """Half model: symmetry plane through the centre stub column, one 20 ft span."""
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)
    sb, sc = W[p["beam"]], W[p["col"]]
    L, dc = p["L"], sc["d"]

    # Steel: piecewise-linear plasticity capped at the measured ultimate stress, which is the
    # class TN 1669 Sec. 5.2.1 uses ("the engineering stress values were input directly in the beam
    # material models").  A Steel02 bilinear fitted through (eps_u, f_u) is only valid up to
    # eps_u: the RBS fractures at 2.0 eps_u, where Steel02's straight hardening line reaches
    # 687 MPa against a measured f_u of 550 -- a 25 % overshoot that goes straight into the moment.
    # The WUF-B fractures at 0.73 eps_u and is unaffected, which is exactly the pattern of the
    # load error.  Backbone: (f_y, eps_y) -> (f_u, eps_u) -> plateau (necking is a localisation a
    # fibre section cannot represent).  Past eps_u the branch keeps 10 % of the pre-ultimate
    # hardening slope: a flat branch localises the fibre section (the required fracture strain
    # jumps from 0.26 to 0.68 for the same measured failure displacement, which is a numerical
    # artefact, not a material property), while 10 % holds the stress within about 4 % of f_u over
    # the whole strain range the model reaches.  Declared in CALIBRATION.md.
    ey = p["fy"] / ES
    Eh = (p["fu"] - p["fy"]) / (p["esu"] - ey)
    e3 = 0.60
    f3 = p["fu"] + 0.10 * Eh * (e3 - p["esu"])
    ops.uniaxialMaterial("Hysteretic", 11,
                         p["fy"], ey, p["fu"], p["esu"], f3, e3,
                         -p["fy"], -ey, -p["fu"], -p["esu"], -f3, -e3,
                         1.0, 1.0, 0.0, 0.0, 0.0)
    ops.uniaxialMaterial("MinMax", MAT_B, 11, "-min", -1.0, "-max", 1.0)   # cannot fracture
    ops.uniaxialMaterial("MinMax", MAT_BF, 11, "-min", -1.0, "-max", 1.0)
    _wsection(SEC_FULL, sb, MAT_B)

    ops.geomTransf("Corotational", 1)
    ops.beamIntegration("Lobatto", 2, SEC_FULL, 5)

    # ---- beam node chain, centre-column face -> end-column face ----------------------------
    xef = L - dc / 2.0
    xs, secs = [dc / 2.0], []          # xs = node stations; secs = ("elastic"|integration tag)
    if p["rbs"]:
        # [READ] Sec. 2.2.2 radius cut: starts a = 0.55 bf from the column face, b = 0.7 db long,
        # c = 0.25 bf deep at its centre.  The cut is modelled as a central min-width element of
        # the SAME gauge length L_conn = 0.5 db used for the WUF-B connection zone, flanked by two
        # slivers whose flange width follows the circular profile.  The gauge length has to be the
        # same in both models because eps_f is a mesh-dependent quantity (TN 1669 Sec. 5.2.1), and
        # transferring eps_f from one specimen to the other is only meaningful at equal gauge.
        a, Lw, c = p["a_fac"] * sb["bf"], p["b_fac"] * sb["d"], p["c_fac"] * sb["bf"]
        R = (4.0 * c ** 2 + Lw ** 2) / (8.0 * c)                   # radius of the cut
        Lg = p["L_conn"]                                           # fracture gauge length
        t = 0.5 * (Lw - Lg)                                        # sliver each side of the gauge
        s_out = 0.5 * (Lg + t)                                     # sliver centre, from cut centre
        cut_out = c - (R - np.sqrt(R ** 2 - s_out ** 2))
        _wsection(SEC_RBS, sb, MAT_BF, bf=sb["bf"] - 2.0 * c)
        _wsection(SEC_RBS_OUT, sb, MAT_BF, bf=sb["bf"] - 2.0 * cut_out)
        ops.beamIntegration("Lobatto", 1, SEC_RBS, 5)
        ops.beamIntegration("Lobatto", 4, SEC_RBS_OUT, 5)
        x0 = dc / 2.0 + a
        xs += [x0, x0 + t, x0 + t + Lg, x0 + Lw]
        secs = ["elastic", 4, 1, 4]
    else:
        # weak zone at the centre-column face.  The test's WUF-B failure is bolt shear followed by
        # bottom-flange fracture; a fibre beam element carries no bolt shear, so the bolts are
        # folded into eps_f of this zone (declared, and its consequence measured in CALIBRATION.md).
        Lw = p["L_conn"]
        _wsection(SEC_RBS, sb, MAT_BF)
        ops.beamIntegration("Lobatto", 1, SEC_RBS, 5)
        xs += [dc / 2.0 + Lw]
        secs = [1]
    xs += [xef - p["L_conn"], xef]
    secs += ["elastic", 2]

    nodes = [NCF]
    ops.node(NCF, xs[0], 0.0)
    for k, x in enumerate(xs[1:], start=1):
        ops.node(30 + k, x, 0.0)
        nodes.append(30 + k)
    ops.node(NCEN, 0.0, 0.0); ops.node(NEC, L, 0.0)
    ops.node(NBASE, L, -p["h_below"]); ops.node(NTOP, L, p["h_col"] - p["h_below"])
    ops.node(NGND, L, p["h_col"] - p["h_below"])

    for k, sec in enumerate(secs):
        na, nb = nodes[k], nodes[k + 1]
        assert xs[k + 1] > xs[k] + 1.0, f"beam nodes out of order at x = {xs[k]:.0f} mm"
        if sec == "elastic":
            ops.element("elasticBeamColumn", 210 + k, na, nb, sb["A"], ES, sb["I"], 1)
        else:
            tag = EWEAK if sec == 1 else 210 + k
            ops.element("forceBeamColumn", tag, na, nb, 1, sec, "-iter", 30, 1e-9)

    # Rigid offset from the column centreline to the column face, then a panel-zone shear spring.
    # t_pz = column web, plus the doubler plate where the specimen has one (TN 1669 Sec. 2.2.2:
    # "Doubler plates were required only for the center column", RBS specimen only).
    rigid = (1e7, ES * 100.0, 1e12)
    ops.node(NPZC, xs[0], 0.0); ops.node(NPZE, xef, 0.0)
    ops.element("elasticBeamColumn", 101, NCEN, NPZC, *rigid, 1)
    ops.element("elasticBeamColumn", 102, NPZE, NEC, *rigid, 1)
    _panel_spring(501, NPZC, NCF, sc, sb, sc["tw"] + p.get("t_doubler", 0.0), p["fy_col"])
    _panel_spring(502, nodes[-1], NPZE, sc, sb, sc["tw"], p["fy_col"])
    # columns stayed elastic in both tests (measured strains below yield)
    ops.element("elasticBeamColumn", 300, NBASE, NEC, sc["A"], ES, sc["I"], 1)
    ops.element("elasticBeamColumn", 301, NEC, NTOP, sc["A"], ES, sc["I"], 1)

    ops.fix(NBASE, 1, 1, 0 if p.get("pin_base") else 1)     # [READ] NIST bracketed fixed / pinned
    ops.fix(NGND, 1, 1, 1)
    ops.fix(NCEN, 1, 0, 1)                                  # symmetry at the centre stub column
    # The two diagonal braces at the top of each end column are replaced by ONE horizontal spring
    # (NIST also replaced them by springs).  Vertical equilibrium then reads exactly
    # P_total = 2 * (base reaction), which is what the model reports as its resistance.
    ops.uniaxialMaterial("Elastic", 40, p["K_br"])
    ops.element("zeroLength", 400, NGND, NTOP, "-mat", 40, "-dir", 1)
    return NCEN


def pushdown(p, verbose=False):
    """Pushdown of the centre column. Returns (u, P, N_beam, dx_endcol) in mm and N."""
    p = dict(p)
    cen = build(p)
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(cen, 0.0, -1.0, 0.0)
    ops.system("UmfPack"); ops.numberer("RCM"); ops.constraints("Transformation")
    ops.test("NormDispIncr", 1e-6, 30, 0); ops.algorithm("Newton")
    ops.integrator("DisplacementControl", cen, 2, -p["du"])
    ops.analysis("Static")

    na, nb = ops.eleNodes(EWEAK)
    # Node standing in for the inclinometers, which were attached to the beam web near the column
    # flange.  One rule, taken from the report's own description of where the hinge forms: for the
    # WUF-B the hinge is AT the column face, so any gauge on the beam sits BEYOND it (node nb); for
    # the RBS the hinge forms in the middle of the cut, and Sec. 4.3.3 says the face inclinometers
    # "do not represent the overall rotation of the beam ... resulting in kinking away from the
    # face", so the gauge sits BEFORE the hinge (node na).
    n_incl = na if p["rbs"] else nb
    u, P, N, dx, th, ri, du = [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], p["du"]
    guard = 0
    while u[-1] < p["u_max"] and guard < 40000:
        guard += 1
        ok = ops.analyze(1)
        if ok != 0:
            for alg, tol, it, f in (("Newton", 1e-4, 60, 1.0), ("ModifiedNewton", 1e-4, 200, 0.5),
                                    ("KrylovNewton", 1e-3, 300, 0.25),
                                    ("NewtonLineSearch", 1e-3, 300, 0.1)):
                ops.test("NormDispIncr", tol, it, 0); ops.algorithm(alg)
                ops.integrator("DisplacementControl", cen, 2, -du * f)
                ok = ops.analyze(1)
                if ok == 0:
                    break
            ops.test("NormDispIncr", 1e-6, 30, 0); ops.algorithm("Newton")
            ops.integrator("DisplacementControl", cen, 2, -du)
            if ok != 0:
                if verbose:
                    print(f"    no convergence at u = {u[-1]:.1f} mm")
                break
        # Resistance from EQUILIBRIUM (base reactions), cross-checked against the load factor:
        # a loose-tolerance fallback step can converge onto a state whose load factor is stale.
        ops.reactions()
        R = 2.0 * ops.nodeReaction(NBASE, 2)
        lam = 2.0 * ops.getLoadFactor(1)
        if abs(R - lam) > 0.02 * max(abs(lam), 1.0e3):
            if verbose:
                print(f"    equilibrium check failed at u = {-ops.nodeDisp(cen, 2):.1f} mm "
                      f"({R/1e3:.1f} kN reactions vs {lam/1e3:.1f} kN load factor)")
            break
        u.append(-ops.nodeDisp(cen, 2)); P.append(R)
        if len(P) > 20 and R < 0.3 * max(P):
            break
        N.append(-ops.eleForce(EWEAK, 1))          # beam axial force, tension positive
        dx.append(-ops.nodeDisp(NEC, 1))           # end-column inward movement at beam level
        # Fracture criterion: the ROTATION across the connection zone, not a fibre strain.  With
        # the displacement imposed and everything outside the gauge elastic, the hinge rotation is
        # kinematic and well conditioned; the extreme-fibre strain is not, because at a plastic
        # plateau the neutral-axis depth -- and hence the strain at a given rotation -- is nearly
        # indeterminate.  Measured: the same failure point needs eps_f = 0.26 / 0.47 / 0.68 for
        # post-ultimate hardening slopes of 100 % / 10 % / 0 %, while the load moves only 1.21 /
        # 1.14 / 1.12.  Rotation is also the criterion the frame evaluator uses.
        th.append(abs(ops.nodeDisp(nb, 3) - ops.nodeDisp(na, 3)))
        ri.append(abs(ops.nodeDisp(n_incl, 3)))
        if th[-1] >= p["theta_u"]:
            break
    return np.array(u), np.array(P), np.array(N), np.array(dx), np.array(th), np.array(ri)


def metrics(u, P, N, dx, th, ri):
    """Response quantities the test reports."""
    out = {}
    if len(u) < 10:
        return out
    k = int(np.argmax(P))
    out.update(P_pk=float(P[k]), u_pk=float(u[k]), N_pk=float(N[k]), dx_pk=float(dx[k]))
    m = u <= 10.0
    K0 = float(np.polyfit(u[m], P[m], 1)[0]) if m.sum() > 3 else float("nan")
    out["K0"] = K0
    # elastic limit: first departure of more than 5 % from the initial tangent line -- this is
    # the report's "remained in the elastic range up to about 2 in", which is a read off a figure,
    # so the 5 % definition is the model's, not the source's, and the comparison is weak.
    iel = next((i for i in range(5, len(u)) if P[i] < 0.95 * K0 * u[i]), None)
    out["u_el"] = float(u[iel]) if iel else float("nan")
    # fracture: first loss of more than 20 % of the running maximum within a 15 mm window.
    # The run stops once the resistance has collapsed, so a fracture in the last few steps is
    # caught by the second test instead of the window.
    for i in range(5, len(P)):
        j = int(np.searchsorted(u, u[i] + 15.0))
        if P[:i].max() > 0 and (P[:i].max() - P[i:j + 1].min()) > 0.20 * P[:i].max():
            out["u_fr"] = float(u[i]); break
    if "u_fr" not in out and P[-1] < 0.5 * P.max():
        out["u_fr"] = float(u[int(np.argmax(P))])
    if "u_fr" not in out:
        out["u_fr"] = float(u[-1])          # the run ends at the rotation criterion
    out["theta_hinge"] = float(th[-1]); out["rot_incl"] = float(ri[-1])
    out["u_end"] = float(u[-1])
    return out


def demo():
    for p in (WUFB, RBS):
        u, P, N, dx, th, ri = pushdown(p, verbose=True)
        assert len(u) > 20, f"{p['name']}: analysis died after {len(u)} steps"
        assert P.max() > 0, f"{p['name']}: no resistance developed"
        m = metrics(u, P, N, dx, th, ri)
        t = TEST[p["name"]]
        print(f"{p['name']:6s} peak {m['P_pk']/1e3:7.1f} kN @ {m['u_pk']:6.1f} mm "
              f"(test {t['P_pk']/1e3:.0f} @ {t['u_pk']:.0f})  N {m['N_pk']/1e3:7.1f} kN "
              f"(test {t['N_beam']/1e3:.0f})  u_el {m['u_el']:5.1f}  u_end {m['u_end']:6.1f}")


if __name__ == "__main__":
    demo()
