"""Xiao, Kunnath, Li, Zhao, Lew & Bao (2015), ACI Struct. J. 112(4) 429-438: half-scale 3-storey,
3 x 3-bay RC frame BUILDING (with slabs) subjected to SUDDEN first-storey column removals by gas
cannon.  Three-dimensional OpenSees model for the building-level calibration of the Ex14-FE
modelling recipe (the connection level is ex14pc/CALIBRATION.md, the planar-frame level is
ex14fe/yi2008_frame.py).

Test phases (paper, Fig. 1(a)):
  I-1   remove corner column A1            (20 kN/m2 on Test-1 area)   A1: 6.3 mm (Table 4)
  I-2   then remove B1                                                  A1 +20.6, B1 +11.0 mm (peaks)
  II-1  remove side column D3              (20 kN/m2 on Test-2 area)   D3: 12 mm (Fig. 6a)
  II-2  then remove D2                                                  D2 +60.8, D3 +35.8 mm (peaks)
        ... the frame then crept to collapse in 14 min under the constant 20 kN/m2
  III   remove interior column B3 on the frame CUT from the D side      B3: 11.5 mm (immediate)

Units N, mm, s; mass in tonne (N s^2/mm).
"""
import math
import time
import numpy as np
import openseespy.opensees as ops

# ---------------------------------------------------------------------------------------------
# geometry  [READ] Fig. 1
# ---------------------------------------------------------------------------------------------
X = [0.0, 4570.0, 9140.0, 13710.0]                  # axes 1..4
Y = [0.0, 3050.0, 6100.0, 9150.0]                   # axes A..D
Z = [0.0, 2280.0, 4110.0, 5940.0]                   # ground, 2nd, 3rd, 4th floor
AX = {"1": 0, "2": 1, "3": 2, "4": 3}
AY = {"A": 0, "B": 1, "C": 2, "D": 3}
NXB, NYB = 4, 3                                     # shell elements per bay in x / y
T_SLAB = 115.0                                      # [READ]
LOAD_AREAS = {                                      # [READ] Fig. 1(a) shading, 20 kN/m2   (x0,x1,y0,y1)
    "test1": (0.0, 2285.0, 0.0, 4575.0),
    # Test 2 is a T (Fig. 1(a) at 600 dpi; the paper's yield-line load area, Eq. (1); Fig. 3(a) of Shan,
    # Petrone & Kunnath 2019): the full central panel C2-C3-D3-D2 plus half-bay strips along D either side
    "test2": [(4570.0, 9140.0, 6100.0, 9150.0), (2285.0, 4570.0, 7625.0, 9150.0), (9140.0, 11425.0, 7625.0, 9150.0)],
    "test2strip": (2285.0, 11425.0, 7625.0, 9150.0),        # the first (300 dpi) misread: the strip only, 13.9 m2
    "test3": (6855.0, 11425.0, 1525.0, 4575.0),
}
Q_TEST = 0.020                                      # [READ] 20 kN/m2 = 0.020 N/mm2
GAMMA = 2.5e-5                                      # [ASSUME] 25 kN/m3 RC
G = 9810.0

# sections  [READ] Tables 1, 2
BEAM_BIG = dict(b=225.0, h=355.0, top_end=4, bot_end=3, top_mid=3, bot_mid=7, dbar=12.0)   # axes 1,4,A,D (+B,C: plan labels 225x355)
BEAM_SMALL = dict(b=200.0, h=280.0, top_end=3, bot_end=3, top_mid=3, bot_mid=3, dbar=12.0, dbot_end=10.0, dtop_mid=10.0)  # axes 2,3
COL_SIZE = {"A1": 350, "A4": 350, "D1": 350, "D4": 350, "B2": 350, "B3": 350, "C2": 350, "C3": 350,
            "A2": 330, "A3": 330, "D2": 330, "D3": 330, "B1": 300, "B4": 300, "C1": 300, "C4": 300}
COL_NBAR = {350: 20, 330: 16, 300: 16}
D_COL_BAR = 14.0
COVER = 20.0                                        # [ASSUME] clear cover, half scale
SLAB_BAR_A = math.pi * 4.0 ** 2 / 200.0             # phi8 @ 200 both ways, two layers -> mm2/mm

# materials  [READ] Table 1-3 footnotes
FY = {10: 432.7, 12: 485.8, 14: 456.8, 8: 453.0}
FU = {10: 649.1, 12: 640.7, 14: 627.4, 8: 657.3}
ES = 200000.0
ESU = 0.10                                          # [ASSUME] elongation not stated; fracture only matters in the collapse pushdown
FCU = dict(col1=41.6, floor1=37.2, col2=42.3, floor2=42.3, col3=36.7, floor3=36.7)   # cube, MPa
FC_FACTOR = 0.8                                     # [READ] "typical conversion factor of 0.8" (paper)
EC_MODE = "GB"                                      # "GB": 1e5/(2.2+34.7/fcu); "ACI": 4700 sqrt(fc)
EC_MULT = 1.0
FT_MULT = 1.0
ETS_DIV = 20.0                                      # Concrete02 tension-softening modulus E_c / ETS_DIV
CONC = "02"                                         # beam/column concrete law: "02" Concrete02 (Kent-Park parabola, tangent 0.6 E_c at
                                                    # 0.35 f_c) or "04" Concrete04 (Popovics; near-linear to ~0.4 f_c, tension exp. decay)
ZETA = 0.05                                         # [ASSUME] Rayleigh damping ratio
F_DAMP = (4.0, 40.0)                                # Hz, Rayleigh anchor frequencies
T_RELEASE = 0.25                                    # [CAL] column-force release time: the measured traces overshoot their settled value by only
                                                    # 5-10 % (Figs. 3, 6a); an instantaneous release gives 40-60 % here.  0.25 s reproduces 1.1
T_FREE = 0.8                                        # s of free vibration simulated after each removal (peaks occur within 0.3 s)
DT = 1.0e-3
TRANSF = "PDelta"                                  # Corotational-3D ignores -jntOffset; PDelta keeps the rigid zones
SLAB_OPTS = ("-implex",)                            # ASDConcrete3D IMPL-EX integration (robust in dynamics)
SLAB_TS = (0.001, 0.4)                              # [CAL] slab tension-stiffening knee (strain, stress/f_t): CEB-FIP stabilised cracking keeps
                                                    # ~0.4 f_t; (0.002, 0.7) -> II-2 45/21 mm, (0.0005, 0.2) -> 93/68, test 60.8/35.8.  None = auto law
COL_SUB = 2                                         # column sub-elements per storey
LOOSE_BLOCKS = False                                # the 20 kN/m2 loading blocks as separate masses on compression-only
                                                    # contacts (they separate when the floor accelerates down faster than g)
K_CONTACT = 2.0e5                                   # N/mm per shell node, block-to-slab contact stiffness
DEBRIS_C = 0.0                                      # [CAL] N s/mm, viscous dashpot at the removed column (the shattered block
                                                    # resisting while it is crushed); 0 = none
DEBRIS_U = 1.0e9                                    # mm of joint drop after which the debris is crushed away (dashpot removed)
SNAP = 0.0                                          # [CAL] fraction of the column force still carried at the end of the ramp and
                                                    # then released within 2 ms (the block's final rupture); 0 = pure ramp
T_SNAP = None                                       # [CAL] s after the start of the release at which the rupture (SNAP -> 0) and the
                                                    # spring-back pulse occur; None = at the end of the ramp (T_RELEASE)
PULSE = 0.0                                         # [CAL] the rupture's spring-back: an UPWARD push on the joint of PULSE x the column
PULSE_T = 0.01                                      # force for PULSE_T seconds right after the rupture; 0 = none
DEBRIS_GAP = 0.0                                    # [CAL] mm: the joint lands on the debris pile after this drop (0 = no contact)
DEBRIS_K = 1.0e5                                    # N/mm contact stiffness of the pile
DEBRIS_FY = 1.0e9                                   # N crushing strength of the pile (elastic-perfectly-plastic contact)
GJ_FAC = 0.5                                        # [ASSUME] cracked torsional stiffness fraction
ELASTIC = False                                     # linear-elastic materials (check against the paper's SAP2000 statics)
ELASTIC_BEAM = False                                # component study: linear-elastic beams only
ELASTIC_COL = False                                 #                  linear-elastic columns only
ELASTIC_SLAB = False                                #                  linear-elastic slab only
OFFSETS = True                                      # rigid joint zones: beam end elements start at the column face (PDelta only; Corotational-3D ignores -jntOffset)
SLAB = True                                         # False = frame only (slab weight kept)
SLAB_E_MULT = 1.0
SHELL = "ShellMITC4"

TEST = dict(A1_I1=6.30, A1_I2=20.60, B1_I2=11.00, D3_II1=17.0, D2_II2=60.8, D3_II2=35.8, B3_III=11.5,
            LF_II=1.0)                                # LF_II: static capacity / 20 kN/m2 (the frame crept to collapse at 1.0)
# D3_II1: the Fig. 6(a) transducer "did not function properly" and only settled (Shan, Petrone & Kunnath 2019,
# Eng Struct 183, Fig. 4a: "Disp. slowly drops to 17 mm"; the paper's own text: "observed 18.5 mm"); scored
# on the model's SETTLED value (0.8 s), not its peak.


def dynamic_set():
    """The 'uncracked-dynamic' surrogate set D of CALIBRATION_BUILDING.md (sudden single removals only):
    dynamic modulus, Popovics compression, an effective cracking strength of 3 f_t (modulus of rupture x
    MC90 rate factor x Branson tension stiffening -- a surrogate, see the record), crack-band softening."""
    global EC_MULT, CONC, FT_MULT, ETS_DIV
    EC_MULT, CONC, FT_MULT, ETS_DIV = 1.25, "04", 3.0, 2.5


def elastic_dynamic_set(column="A1"):
    """Set E of CALIBRATION_BUILDING.md -- the model of the ELASTIC single-removal regime: linear-elastic
    materials at a dynamic modulus E_d = 1.40 E_c (45 GPa; BS 8110-2 gives 41 -- the extra 10 % is [CAL] on
    the equilibrium the A1 trace rings about, 5.7 mm) and the shot-out column block fitted per column on the
    measured trace.
    A1: the block is crushed over 80 ms (95 % of its force released on a ramp) while resisting as a dashpot
    of 900 N s/mm until the joint has dropped 5.3 mm; at 0.13 s the remaining 5 % ruptures and the fractured
    block springs back -- an upward push of 20 % of the column force for 20 ms; the floor then rings freely
    at its 8-Hz corner mode.  Test / model: descent -1.5 -> -5.7 mm in 0.06 / 0.068 s; first valley 5.7 at
    0.07 s / 5.85 at 0.09; first peak 5.1 at 0.13 / 5.5 at 0.11; second valley 6.0 / 6.6; ringing 8.1 Hz
    +-0.43 / 7.9 Hz +-0.73; settled 5.7 -> 6.0 (creep) / 5.56.
    D3: 20-ms release with a dashpot of 600 N s/mm.  Not for the two-column loss of phase II (use R2)."""
    global ELASTIC, EC_MULT, T_RELEASE, SNAP, T_SNAP, PULSE, PULSE_T, DEBRIS_C, DEBRIS_U, DEBRIS_GAP
    ELASTIC, EC_MULT, DEBRIS_GAP = True, 1.40, 0.0
    if column == "A1":
        T_RELEASE, SNAP, T_SNAP, PULSE, PULSE_T, DEBRIS_C, DEBRIS_U = 0.08, 0.05, 0.13, 0.20, 0.02, 900.0, 5.3
    else:
        T_RELEASE, SNAP, T_SNAP, PULSE, DEBRIS_C, DEBRIS_U = 0.02, 0.0, None, 0.0, 600.0, 1.0e9


# ---------------------------------------------------------------------------------------------
def _ec(fcu):
    fc = FC_FACTOR * fcu
    if EC_MODE == "GB":
        return EC_MULT * 1.0e5 / (2.2 + 34.7 / fcu)
    return EC_MULT * 4700.0 * math.sqrt(fc)


def _ft(fcu):
    return FT_MULT * 0.395 * fcu ** 0.55            # GB 50010


_mat = {}


def _materials():
    """Uniaxial + nD materials.  Tags in _mat."""
    t = 1
    for d in (8, 10, 12, 14):
        ops.uniaxialMaterial("Steel02", t, FY[d], ES, (FU[d] - FY[d]) / (ESU * ES), 18.0, 0.925, 0.15)
        _mat[("steel", d)] = t; t += 1
        ops.uniaxialMaterial("MinMax", t, t - 1, "-min", -ESU, "-max", ESU)
        _mat[("steel_mm", d)] = t; t += 1
    for key, fcu in FCU.items():
        fc, ec = FC_FACTOR * fcu, _ec(fcu)
        eps0 = 2.0 * fc / ec
        if CONC == "04":
            # Popovics compression with the explicit E_c; tension linear to f_t, exponential decay (beta 0.1)
            # to zero at et = 20 f_t / E_c (the same end strain as the Concrete02 law with ETS_DIV 20)
            ft_ = _ft(fcu); et = ft_ / ec * (1.0 + ETS_DIV)
            ops.uniaxialMaterial("Concrete04", t, -fc, -eps0, -0.006, ec, ft_, et, 0.1)
            _mat[("cover", key)] = t; t += 1
            ops.uniaxialMaterial("Concrete04", t, -1.15 * fc, -1.15 * eps0 * 1.5, -0.02, ec, ft_, et, 0.1)
            _mat[("core", key)] = t; t += 1
        else:
            # cover / unconfined
            ops.uniaxialMaterial("Concrete02", t, -fc, -eps0, -0.2 * fc, -0.006, 0.1, _ft(fcu), ec / ETS_DIV)
            _mat[("cover", key)] = t; t += 1
            # core: light confinement (phi8 ties @125 / stirrups @150-225) -> k = 1.15  [ASSUME]
            ops.uniaxialMaterial("Concrete02", t, -1.15 * fc, -1.15 * eps0 * 1.5, -0.3 * fc, -0.02, 0.1, _ft(fcu), ec / ETS_DIV)
            _mat[("core", key)] = t; t += 1
        if key.startswith("floor"):
            # slab concrete: plane-stress damage model (Lu et al.), fc, ft, fcu(residual), eps_c0, eps_cu, eps_tu, stc
            # slab concrete: ASDConcrete3D (Petracca) damage model with the auto-generated backbones
            # from f_c, f_t; PlateFiber wrapper enforces sigma_zz = 0 for the shell layers
            ft = _ft(fcu)
            if SLAB_TS is None:
                ops.nDMaterial("ASDConcrete3D", t, ec, 0.2, "-fc", fc, "-ft", ft, *SLAB_OPTS)
            else:
                # explicit backbones.  Tension: linear to f_t, then tension stiffening down to
                # SLAB_TS[1] f_t at strain SLAB_TS[0] (smeared, reinforced), residual 0.05 f_t
                Te = [0.0, ft / ec, SLAB_TS[0], 0.02]
                Ts = [0.0, ft, SLAB_TS[1] * ft, 0.05 * ft]
                Td = [0.0, 0.0] + [1.0 - Ts[i] / (ec * Te[i]) for i in (2, 3)]
                e0 = 2.0 * fc / ec
                Ce = [0.0, 0.4 * fc / ec, e0, 0.0035, 0.01]
                Cs = [0.0, 0.4 * fc, fc, 0.6 * fc, 0.2 * fc]
                Cd = [0.0, 0.0] + [1.0 - Cs[i] / (ec * Ce[i]) for i in (2, 3, 4)]
                ops.nDMaterial("ASDConcrete3D", t, ec, 0.2, "-Te", *Te, "-Ts", *Ts, "-Td", *Td,
                               "-Ce", *Ce, "-Cs", *Cs, "-Cd", *Cd, *SLAB_OPTS)
            ops.nDMaterial("PlateFiber", t + 1, t)
            _mat[("slab", key)] = t + 1; t += 2
    ops.nDMaterial("PlateRebar", t, _mat[("steel", 8)], 0.0); _mat["rebar_x"] = t; t += 1
    ops.nDMaterial("PlateRebar", t, _mat[("steel", 8)], 90.0); _mat["rebar_y"] = t; t += 1
    # elastic torsion for the 3D fibre sections
    ops.uniaxialMaterial("Elastic", t, 1.0e12); _mat["GJ"] = t; t += 1
    _mat["_next"] = t
    if ELASTIC_BEAM or ELASTIC_COL or ELASTIC_SLAB:
        t = 600
        ops.uniaxialMaterial("Elastic", t, ES); es_tag = t; t += 1
        for key, fcu in FCU.items():
            if (key.startswith("floor") and ELASTIC_BEAM) or (key.startswith("col") and ELASTIC_COL):
                ops.uniaxialMaterial("Elastic", t, _ec(fcu)); _mat[("cover", key)] = t; _mat[("core", key)] = t; t += 1
            if key.startswith("floor") and ELASTIC_SLAB:
                ops.nDMaterial("ElasticIsotropic", t, _ec(fcu), 0.2)
                ops.nDMaterial("PlateFiber", t + 1, t); _mat[("slab", key)] = t + 1; t += 2
        for d in ((10, 12) if ELASTIC_BEAM else ()) + ((14,) if ELASTIC_COL else ()):
            _mat[("steel_mm", d)] = es_tag
        if ELASTIC_SLAB:
            ops.nDMaterial("PlateRebar", t, es_tag, 0.0); _mat["rebar_x"] = t; t += 1
            ops.nDMaterial("PlateRebar", t, es_tag, 90.0); _mat["rebar_y"] = t; t += 1
    if ELASTIC:
        _mat.clear(); t = 500
        ops.uniaxialMaterial("Elastic", t, ES)
        for d in (8, 10, 12, 14):
            _mat[("steel", d)] = t; _mat[("steel_mm", d)] = t
        t += 1
        for key, fcu in FCU.items():
            ops.uniaxialMaterial("Elastic", t, _ec(fcu)); _mat[("cover", key)] = t; _mat[("core", key)] = t; t += 1
            if key.startswith("floor"):
                ops.nDMaterial("ElasticIsotropic", t, SLAB_E_MULT * _ec(fcu), 0.2)
                ops.nDMaterial("PlateFiber", t + 1, t); _mat[("slab", key)] = t + 1; t += 2
        ops.nDMaterial("PlateRebar", t, _mat[("steel", 8)], 0.0); _mat["rebar_x"] = t; t += 1
        ops.nDMaterial("PlateRebar", t, _mat[("steel", 8)], 90.0); _mat["rebar_y"] = t; t += 1


def _slab_section(tag, key):
    c = _mat[("slab", key)]
    cov = 15.0                                       # [ASSUME] slab cover
    tb = SLAB_BAR_A                                  # equivalent bar-layer thickness, each direction, each face
    core = T_SLAB - 2.0 * cov - 4.0 * tb
    layers = [c, cov, _mat["rebar_x"], tb, _mat["rebar_y"], tb, c, core,
              _mat["rebar_y"], tb, _mat["rebar_x"], tb, c, cov]
    ops.section("LayeredShell", tag, 7, *layers)


def _gj(b, h, fcu):
    """Elastic torsional stiffness of the rectangle b x h, times GJ_FAC (cracked torsion)."""
    a, c = max(b, h), min(b, h)
    beta = 1.0 / 3.0 - 0.21 * (c / a) * (1.0 - (c / a) ** 4 / 12.0)
    return GJ_FAC * _ec(fcu) / 2.4 * beta * a * c ** 3


def _beam_section(tag, spec, key, at_end, dtop=None, dbot=None):
    """Beam fibre section referenced to the SLAB MID-PLANE node (composite T action through the common
    nodes): concrete web below the slab only (the slab strip over the beam is in the shells)."""
    b, h = spec["b"], spec["h"]
    y_top = T_SLAB / 2.0                             # slab top relative to the node
    y_bot = y_top - h
    y_web = y_top - T_SLAB                           # slab soffit
    cov, cor = _mat[("cover", key)], _mat[("core", key)]
    ops.section("Fiber", tag, "-GJ", _gj(b, h, FCU[key]))
    # web: core inside the stirrups, cover outside
    c = COVER + 8.0
    ops.patch("rect", cor, 8, 4, y_bot + c, -b / 2 + c, y_web, b / 2 - c)
    ops.patch("rect", cov, 8, 1, y_bot + c, -b / 2, y_web, -b / 2 + c)
    ops.patch("rect", cov, 8, 1, y_bot + c, b / 2 - c, y_web, b / 2)
    ops.patch("rect", cov, 2, 4, y_bot, -b / 2, y_bot + c, b / 2)
    n_top = spec["top_end"] if at_end else spec["top_mid"]
    n_bot = spec["bot_end"] if at_end else spec["bot_mid"]
    d_top = dtop or spec["dbar"]
    d_bot = dbot or spec["dbar"]
    a_top, a_bot = math.pi * d_top ** 2 / 4, math.pi * d_bot ** 2 / 4
    st_top, st_bot = _mat[("steel_mm", int(d_top))], _mat[("steel_mm", int(d_bot))]
    zz = b / 2 - COVER - 8.0 - d_top / 2
    ops.layer("straight", st_top, n_top, a_top, y_top - COVER - 8.0 - d_top / 2, -zz, y_top - COVER - 8.0 - d_top / 2, zz)
    yb = y_bot + COVER + 8.0 + d_bot / 2
    if n_bot > 4:                                    # 7 phi12 in two layers  [READ]
        n1, n2 = 4, n_bot - 4
        ops.layer("straight", st_bot, n1, a_bot, yb, -zz, yb, zz)
        ops.layer("straight", st_bot, n2, a_bot, yb + 25.0, -zz * 0.6, yb + 25.0, zz * 0.6)
    else:
        ops.layer("straight", st_bot, n_bot, a_bot, yb, -zz, yb, zz)


def _col_section(tag, size, key):
    cov, cor = _mat[("cover", key)], _mat[("core", key)]
    st = _mat[("steel_mm", 14)]
    s = size / 2.0
    c = COVER + 8.0
    ops.section("Fiber", tag, "-GJ", _gj(size, size, FCU[key]))
    ops.patch("rect", cor, 8, 8, -s + c, -s + c, s - c, s - c)
    ops.patch("rect", cov, 8, 1, -s, -s, s, -s + c)
    ops.patch("rect", cov, 8, 1, -s, s - c, s, s)
    ops.patch("rect", cov, 1, 8, -s, -s + c, -s + c, s - c)
    ops.patch("rect", cov, 1, 8, s - c, -s + c, s, s - c)
    n = COL_NBAR[size]
    a = math.pi * D_COL_BAR ** 2 / 4
    r = s - c - D_COL_BAR / 2
    per_side = n // 4 + 1                            # bars per side incl. corners
    ops.layer("straight", st, per_side, a, -r, -r, r, -r)
    ops.layer("straight", st, per_side, a, -r, r, r, r)
    ops.layer("straight", st, per_side - 2, a, -r, -r + 2 * r / (per_side - 1), -r, r - 2 * r / (per_side - 1))
    ops.layer("straight", st, per_side - 2, a, r, -r + 2 * r / (per_side - 1), r, r - 2 * r / (per_side - 1))


# ---------------------------------------------------------------------------------------------
NODE = {}          # (lvl, ix, iy) -> tag
COL_ELE = {}       # "A1" -> [ele tags storey 1 (bottom..top)]
COL_MID = {}       # "A1" -> [intermediate node tags of storey 1]
LOADED = {}        # node tag -> load (N, downward positive)
BLOCK = {}         # node tag -> block load (N), separated when LOOSE_BLOCKS
MASS = {}
MON = {}           # name -> node tag
BEAM_ELE = {}      # (lvl, "x"|"y", line k, bay, j) -> ele tag


def _xs():
    xs = [X[0]]
    for i in range(3):
        xs += list(np.linspace(X[i], X[i + 1], NXB + 1)[1:])
    ys = [Y[0]]
    for i in range(3):
        ys += list(np.linspace(Y[i], Y[i + 1], NYB + 1)[1:])
    return xs, ys


def build(cut_D=False, remove_beams_CD=False):
    """cut_D: Phase III condition -- beams and slabs between axes C and D removed at every floor
    (the paper: 'the frame was separated at each story level near Frame C')."""
    ops.wipe(); ops.model("basic", "-ndm", 3, "-ndf", 6)
    NODE.clear(); COL_ELE.clear(); COL_MID.clear(); BEAM_ELE.clear(); BLOCK.clear(); LOADED.clear(); MASS.clear(); MON.clear(); _mat.clear()
    _materials()
    xs, ys = _xs()
    tag = 1
    for lvl in range(4):
        for ix, x in enumerate(xs):
            for iy, y in enumerate(ys):
                if lvl == 0 and not (ix % NXB == 0 and iy % NYB == 0):
                    continue
                if cut_D and lvl > 0 and iy > 2 * NYB:      # nodes strictly beyond axis C dropped
                    continue
                ops.node(tag, x, y, Z[lvl]); NODE[(lvl, ix, iy)] = tag
                if lvl == 0:
                    ops.fix(tag, 1, 1, 1, 1, 1, 1)
                tag += 1
    # transformations
    ops.geomTransf(TRANSF, 1, 0.0, -1.0, 0.0)        # beams along x: local y = global z (up)
    ops.geomTransf(TRANSF, 2, 1.0, 0.0, 0.0)         # beams along y
    ops.geomTransf("PDelta", 3, 1.0, 0.0, 0.0)       # columns
    # sections
    sec = {}
    stag = 1
    for lvl in (1, 2, 3):
        key = f"floor{lvl}"
        _slab_section(stag, key); sec[("slab", lvl)] = stag; stag += 1
        for name, spec in (("big", BEAM_BIG), ("small", BEAM_SMALL)):
            for at_end in (True, False):
                kw = {}
                if name == "small":
                    kw = dict(dbot=spec["dbot_end"]) if at_end else dict(dtop=spec["dtop_mid"])
                _beam_section(stag, spec, key, at_end, **kw); sec[(name, at_end, lvl)] = stag; stag += 1
    for s in (1, 2, 3):
        for size in (350, 330, 300):
            _col_section(stag, size, f"col{s}"); sec[("col", size, s)] = stag; stag += 1
    ops.beamIntegration("Lobatto", 1, 1, 3)          # placeholder, replaced per element below
    itag = 100
    ele = 1
    # ---- columns
    for cname, size in COL_SIZE.items():
        ix, iy = AX[cname[1]] * NXB, AY[cname[0]] * NYB
        if cut_D and iy > 2 * NYB:
            continue
        COL_ELE[cname] = []; COL_MID[cname] = []
        for s in (1, 2, 3):
            nb, nt = NODE[(s - 1, ix, iy)], NODE[(s, ix, iy)]
            zb, zt = Z[s - 1], Z[s]
            ops.beamIntegration("Lobatto", itag, sec[("col", size, s)], 3)
            prev = nb
            for k in range(1, COL_SUB + 1):
                if k < COL_SUB:
                    ops.node(tag, xs[ix], ys[iy], zb + (zt - zb) * k / COL_SUB); nxt = tag; tag += 1
                    if s == 1:
                        COL_MID[cname].append(nxt)
                else:
                    nxt = nt
                ctr = 3
                if OFFSETS:
                    # rigid joint zone inside the beam depth: from the node (slab mid-plane) up to the slab
                    # top at the column's lower end, and down to the beam soffit at its upper end
                    dI = (0.0, 0.0, T_SLAB / 2.0) if (k == 1 and s > 1) else (0.0, 0.0, 0.0)
                    dJ = (0.0, 0.0, -(BEAM_BIG["h"] - T_SLAB / 2.0)) if k == COL_SUB else (0.0, 0.0, 0.0)
                    ops.geomTransf("PDelta", 3000 + ele, 1.0, 0.0, 0.0, "-jntOffset", *dI, *dJ)
                    ctr = 3000 + ele
                ops.element("dispBeamColumn", ele, prev, nxt, ctr, itag)
                if s == 1:
                    COL_ELE[cname].append(ele)
                ele += 1; prev = nxt
            itag += 1
    # ---- beams (x direction on axes A..D, y direction on axes 1..4)
    def beam_line(lvl, axis_is_x, k):
        name = "big" if (axis_is_x or k in (0, 3)) else "small"
        spec = BEAM_BIG if name == "big" else BEAM_SMALL
        w = spec["b"] * (spec["h"] - T_SLAB) * GAMMA          # web self-weight per mm
        n_sub, pts = (NXB, xs) if axis_is_x else (NYB, ys)
        L_bay = (X[1] - X[0]) if axis_is_x else (Y[1] - Y[0])
        for bay in range(3):
            for j in range(n_sub):
                i0, i1 = bay * n_sub + j, bay * n_sub + j + 1
                at_end = (j == 0 or j == n_sub - 1)
                key = (name, at_end, lvl)
                if axis_is_x:
                    a, b = NODE.get((lvl, i0, k * NYB)), NODE.get((lvl, i1, k * NYB))
                else:
                    a, b = NODE.get((lvl, k * NXB, i0)), NODE.get((lvl, k * NXB, i1))
                if a is None or b is None:
                    continue
                off = None
                if at_end:
                    # half column width at the column end(s) of this sub-element
                    def csize(iax, jax):
                        return COL_SIZE.get(("ABCD"[jax] + "1234"[iax]), 0) / 2.0
                    if axis_is_x:
                        ci = csize(bay, k) if j == 0 else 0.0
                        cj = csize(bay + 1, k) if j == n_sub - 1 else 0.0
                        off = (ci, 0.0, 0.0, -cj, 0.0, 0.0)
                    else:
                        ci = csize(k, bay) if j == 0 else 0.0
                        cj = csize(k, bay + 1) if j == n_sub - 1 else 0.0
                        off = (0.0, ci, 0.0, 0.0, -cj, 0.0)
                BEAM_ELE[(lvl, "x" if axis_is_x else "y", k, bay, j)] = \
                    _add_beam(a, b, 1 if axis_is_x else 2, sec[key], w * (pts[i1] - pts[i0]), off)
        return
    def _add_beam(a, b, tr, stag_, wt, off=None):
        nonlocal ele, itag
        ops.beamIntegration("Lobatto", itag, stag_, 3)
        if off is not None and OFFSETS:
            vx = (0.0, -1.0, 0.0) if tr == 1 else (1.0, 0.0, 0.0)
            ops.geomTransf(TRANSF, 1000 + ele, *vx, "-jntOffset", *off)
            tr = 1000 + ele
        ops.element("dispBeamColumn", ele, a, b, tr, itag)
        for n in (a, b):
            LOADED[n] = LOADED.get(n, 0.0) + wt / 2
        ele += 1; itag += 1
        return ele - 1
    for lvl in (1, 2, 3):
        for k in range(4):
            beam_line(lvl, True, k)
            beam_line(lvl, False, k)
    # ---- slabs
    for lvl in (1, 2, 3):
        for ix in range(len(xs) - 1):
            for iy in range(len(ys) - 1):
                n = [NODE.get((lvl, ix, iy)), NODE.get((lvl, ix + 1, iy)),
                     NODE.get((lvl, ix + 1, iy + 1)), NODE.get((lvl, ix, iy + 1))]
                if None in n:
                    continue
                if SLAB:
                    ops.element(SHELL, ele, *n, sec[("slab", lvl)]); ele += 1
                A = (xs[ix + 1] - xs[ix]) * (ys[iy + 1] - ys[iy])
                for t in n:
                    LOADED[t] = LOADED.get(t, 0.0) + GAMMA * T_SLAB * A / 4
    # monitors: column-top nodes at the 2nd floor, mid D2-D3
    for cname in COL_SIZE:
        ix, iy = AX[cname[1]] * NXB, AY[cname[0]] * NYB
        if (1, ix, iy) in NODE:
            MON[cname] = NODE[(1, ix, iy)]
    if (1, int(1.5 * NXB), 3 * NYB) in NODE:
        MON["mid_D23"] = NODE[(1, int(1.5 * NXB), 3 * NYB)]
    return xs, ys


def _in_area(area, cx, cy):
    rects = LOAD_AREAS[area]
    if isinstance(rects, tuple):
        rects = [rects]
    return any(x0 <= cx <= x1 and y0 <= cy <= y1 for x0, x1, y0, y1 in rects)


def add_test_load(area, lvls=(1, 2, 3)):
    """20 kN/m2 on the shaded area at every floor (paper: blocks on the tributary area of the columns
    to be removed; Fig. 1(a) shows one plan -> applied at all three floors  [ASSUME])."""
    xs, ys = _xs()
    for lvl in lvls:
        for ix in range(len(xs) - 1):
            for iy in range(len(ys) - 1):
                cx, cy = 0.5 * (xs[ix] + xs[ix + 1]), 0.5 * (ys[iy] + ys[iy + 1])
                if not _in_area(area, cx, cy):
                    continue
                n = [NODE.get((lvl, ix, iy)), NODE.get((lvl, ix + 1, iy)),
                     NODE.get((lvl, ix + 1, iy + 1)), NODE.get((lvl, ix, iy + 1))]
                if None in n:
                    continue
                A = (xs[ix + 1] - xs[ix]) * (ys[iy + 1] - ys[iy])
                for t in n:
                    if LOOSE_BLOCKS:
                        BLOCK[t] = BLOCK.get(t, 0.0) + Q_TEST * A / 4
                    else:
                        LOADED[t] = LOADED.get(t, 0.0) + Q_TEST * A / 4


BLOCK_ELE = []


def gravity(verbose=False):
    ops.timeSeries("Linear", 1); ops.pattern("Plain", 1, 1)
    for n, w in LOADED.items():
        ops.load(n, 0.0, 0.0, -w, 0.0, 0.0, 0.0)
        ops.mass(n, w / G, w / G, w / G, 0.0, 0.0, 0.0)
    BLOCK_ELE.clear()
    if BLOCK:
        ops.uniaxialMaterial("ENT", 9000, K_CONTACT)
        btag, etag = 900000, 900000
        for n, w in BLOCK.items():
            x, y, z = ops.nodeCoord(n)
            ops.node(btag, x, y, z); ops.fix(btag, 1, 1, 0, 1, 1, 1)
            ops.mass(btag, 0.0, 0.0, w / G, 0.0, 0.0, 0.0)
            ops.load(btag, 0.0, 0.0, -w, 0.0, 0.0, 0.0)
            ops.element("zeroLength", etag, n, btag, "-mat", 9000, "-dir", 3)   # compression only: block sits on the slab
            BLOCK_ELE.append(etag); btag += 1; etag += 1
    ops.system("UmfPack"); ops.numberer("RCM"); ops.constraints("Plain")
    ops.test("NormDispIncr", 1.0e-4, 50, 0); ops.algorithm("Newton")
    ops.integrator("LoadControl", 0.1); ops.analysis("Static")
    for _ in range(10):
        if ops.analyze(1) != 0:
            ops.algorithm("ModifiedNewton"); ops.test("NormDispIncr", 1.0e-3, 200, 0)
            if ops.analyze(1) != 0:
                raise RuntimeError("gravity failed")
            ops.algorithm("Newton"); ops.test("NormDispIncr", 1.0e-4, 50, 0)
    ops.loadConst("-time", 0.0)
    if verbose:
        tot = sum(LOADED.values())
        print(f"  gravity: total {tot/1e3:.0f} kN on {len(LOADED)} nodes")


def bar_strain(lvl, axis, k, bay, j, sec_ip, top=True):
    """Strain of the outermost top (or bottom) bar layer at integration point sec_ip (1..3) of beam
    sub-element (lvl, axis, k, bay, j)."""
    e = BEAM_ELE[(lvl, axis, k, bay, j)]
    big = axis == "x" or k in (0, 3)
    h = (BEAM_BIG if big else BEAM_SMALL)["h"]
    y = T_SLAB / 2 - COVER - 8.0 - 6.0 if top else T_SLAB / 2 - h + COVER + 8.0 + 6.0
    return ops.eleResponse(e, "section", sec_ip, "fiber", y, 0.0, "stressStrain")[1]


def _disp(name, dof=3):
    return ops.nodeDisp(MON[name], dof)


_ptag = [10]


def remove_column(cname, t_now, verbose=False):
    """Replace the first-storey column by its end force, then release it over T_RELEASE (Path series)."""
    top = COL_ELE[cname][-1]
    f = ops.eleForce(top)[6:12]                      # element end forces at the top node (global)
    for e in COL_ELE[cname]:
        ops.remove("element", e)
    for n in COL_MID[cname]:
        ops.remove("node", n)
    _ptag[0] += 1
    ts = t_now + (T_SNAP if T_SNAP is not None else T_RELEASE)     # rupture time (>= end of the ramp)
    if SNAP > 0.0 and PULSE > 0.0:
        ops.timeSeries("Path", _ptag[0], "-time", 0.0, t_now, t_now + T_RELEASE, ts, ts + 0.002, ts + 0.002 + PULSE_T, ts + 0.004 + PULSE_T, 1.0e6,
                       "-values", 1.0, 1.0, SNAP, SNAP, PULSE, PULSE, 0.0, 0.0)
    elif SNAP > 0.0:
        ops.timeSeries("Path", _ptag[0], "-time", 0.0, t_now, t_now + T_RELEASE, ts, ts + 0.002, 1.0e6,
                       "-values", 1.0, 1.0, SNAP, SNAP, 0.0, 0.0)
    else:
        ops.timeSeries("Path", _ptag[0], "-time", 0.0, t_now, t_now + T_RELEASE, 1.0e6,
                       "-values", 1.0, 1.0, 0.0, 0.0)
    ops.pattern("Plain", _ptag[0], _ptag[0])
    ops.load(MON[cname], *[-v for v in f])           # the force the column exerted on the node
    if DEBRIS_C > 0.0:
        x, y, z = ops.nodeCoord(MON[cname])
        nt = 950000 + 10 * _ptag[0]
        ops.node(nt, x, y, z); ops.fix(nt, 1, 1, 1, 1, 1, 1)
        ops.uniaxialMaterial("Viscous", nt, DEBRIS_C, 1.0)
        ops.uniaxialMaterial("MinMax", nt + 1, nt, "-min", -DEBRIS_U, "-max", DEBRIS_U)   # dies once crushed through (either sign)
        ops.element("zeroLength", nt, MON[cname], nt, "-mat", nt + 1, "-dir", 3)
    if DEBRIS_GAP > 0.0:
        x, y, z = ops.nodeCoord(MON[cname])
        ng = 960000 + 10 * _ptag[0]
        ops.node(ng, x, y, z); ops.fix(ng, 1, 1, 1, 1, 1, 1)
        # deformation = u_fixed - u_joint = +drop: the gap closes after DEBRIS_GAP mm and the pile pushes back
        ops.uniaxialMaterial("ElasticPPGap", ng, DEBRIS_K, DEBRIS_FY, DEBRIS_GAP, 0.0, "damage")
        ops.element("zeroLength", ng, MON[cname], ng, "-mat", ng, "-dir", 3)
    if verbose:
        print(f"  {cname}: column top force N = {-f[2]/1e3:.1f} kN (compression), released at t = {t_now:.2f} s")
    return -f[2]


def transient(t_end, monitors, verbose=False, rec=None):
    """Newmark to t_end; returns dict name -> (min disp, disp at end) increments are the caller's job."""
    w1, w2 = 2 * math.pi * F_DAMP[0], 2 * math.pi * F_DAMP[1]
    a0, a1 = 2 * ZETA * w1 * w2 / (w1 + w2), 2 * ZETA / (w1 + w2)
    if BLOCK_ELE:
        # stiffness-proportional damping on the structure only: none on the block contacts
        struct = [e for e in ops.getEleTags() if e < 900000]
        ops.region(1, "-ele", *struct, "-rayleigh", 0.0, 0.0, 0.0, a1)
        ops.region(2, "-node", *ops.getNodeTags(), "-rayleigh", a0, 0.0, 0.0, 0.0)
    else:
        ops.rayleigh(a0, 0.0, 0.0, a1)
    ops.constraints("Plain"); ops.numberer("RCM"); ops.system("UmfPack")
    ops.test("NormDispIncr", 1.0e-4, 30, 0); ops.algorithm("Newton")
    ops.integrator("Newmark", 0.5, 0.25); ops.analysis("Transient")
    peaks = {m: (0.0, 0.0) for m in monitors}
    t = ops.getTime()
    while t < t_end - 1e-9:
        dt = DT
        ok = ops.analyze(1, dt)
        if ok != 0:
            for alg, sub in (("ModifiedNewton", 1), ("KrylovNewton", 1), ("Newton", 10), ("KrylovNewton", 50)):
                ops.algorithm(alg); ops.test("NormDispIncr", 1.0e-3, 200, 0)
                ok = ops.analyze(sub, dt / sub)
                if ok == 0:
                    break
            ops.algorithm("Newton"); ops.test("NormDispIncr", 1.0e-4, 30, 0)
            if ok != 0:
                if verbose:
                    print(f"  transient failed at t = {ops.getTime():.4f} s")
                return peaks, False
        t = ops.getTime()
        for m in monitors:
            d = ops.nodeDisp(MON[m], 3)
            peaks[m] = (min(peaks[m][0], d), d)
        if rec is not None:
            rec.append([t] + [ops.nodeDisp(MON[m], 3) for m in monitors]
                       + [ops.nodeDisp(MON[m], 1) if m in MON else 0.0 for m in ("D1", "D4")])
    return peaks, True


def pushdown(area, u_stop=450.0, du_lf=0.01, verbose=False, mon="D2"):
    """Static load-factor increase of the test-area load on the current (columns removed) state.
    Returns (max load factor reached relative to 20 kN/m2, D2 displacement there)."""
    xs, ys = _xs()
    _ptag[0] += 1
    t_now = ops.getTime()                            # ramp starts from the current domain time
    ops.timeSeries("Path", _ptag[0], "-time", 0.0, t_now, t_now + 1.0e3, "-values", 0.0, 0.0, 1.0e3)
    ops.pattern("Plain", _ptag[0], _ptag[0])
    for lvl in (1, 2, 3):
        for ix in range(len(xs) - 1):
            for iy in range(len(ys) - 1):
                cx, cy = 0.5 * (xs[ix] + xs[ix + 1]), 0.5 * (ys[iy] + ys[iy + 1])
                if not _in_area(area, cx, cy):
                    continue
                n = [NODE.get((lvl, ix, iy)), NODE.get((lvl, ix + 1, iy)),
                     NODE.get((lvl, ix + 1, iy + 1)), NODE.get((lvl, ix, iy + 1))]
                if None in n:
                    continue
                A = (xs[ix + 1] - xs[ix]) * (ys[iy + 1] - ys[iy])
                for t in n:
                    ops.load(t, 0.0, 0.0, -Q_TEST * A / 4, 0.0, 0.0, 0.0)
    ops.rayleigh(0.0, 0.0, 0.0, 0.0)
    ops.constraints("Plain"); ops.numberer("RCM"); ops.system("UmfPack")
    ops.test("NormDispIncr", 1.0e-4, 50, 0); ops.algorithm("Newton")
    ops.integrator("LoadControl", du_lf); ops.analysis("Static")
    lf, hist = 0.0, []
    while True:
        ok = ops.analyze(1)
        if ok != 0:
            ok = -1
            for alg, f in (("ModifiedNewton", 1.0), ("KrylovNewton", 0.5), ("KrylovNewton", 0.1)):
                ops.algorithm(alg); ops.test("NormDispIncr", 1.0e-3, 300, 0)
                ops.integrator("LoadControl", du_lf * f)
                ok = ops.analyze(1)
                if ok == 0:
                    lf += du_lf * f; break
            ops.algorithm("Newton"); ops.test("NormDispIncr", 1.0e-4, 50, 0); ops.integrator("LoadControl", du_lf)
            if ok != 0:
                break
        else:
            lf += du_lf
        d = -_disp(mon)
        hist.append((1.0 + lf, d))
        if verbose and len(hist) % 10 == 0:
            print(f"    LF {1+lf:.2f}  {mon} {d:.0f} mm")
        if d > u_stop:
            break
    return 1.0 + lf, hist


# ---------------------------------------------------------------------------------------------
def phase_I(verbose=True, rec=None, with_pushdown=False):
    build(); add_test_load("test1"); gravity(verbose)
    out = {}
    d0 = _disp("A1")
    remove_column("A1", 0.0, verbose)
    pk, ok = transient(T_FREE, ["A1", "B1"], verbose, rec)
    out["A1_I1"] = d0 - pk["A1"][0]
    a1, b1 = _disp("A1"), _disp("B1")
    remove_column("B1", T_FREE, verbose)
    pk, ok2 = transient(2 * T_FREE, ["A1", "B1"], verbose, rec)
    out["A1_I2"] = a1 - pk["A1"][0]; out["B1_I2"] = b1 - pk["B1"][0]
    out["_ok"] = ok and ok2
    if with_pushdown and out["_ok"]:
        lf, hist = pushdown("test1", verbose=verbose, mon="A1")
        out["LF_I"] = lf; out["_push"] = hist
    return out


def phase_II(verbose=True, rec=None, with_pushdown=True, area="test2"):
    build(); add_test_load(area); gravity(verbose)
    out = {}
    d0 = _disp("D3")
    remove_column("D3", 0.0, verbose)
    pk, ok = transient(T_FREE, ["D2", "D3", "mid_D23"], verbose, rec)
    out["D3_II1_peak"] = d0 - pk["D3"][0]
    out["D3_II1"] = d0 - pk["D3"][1]                 # settled: the test transducer only recorded the settled value
    d2, d3 = _disp("D2"), _disp("D3")
    remove_column("D2", T_FREE, verbose)
    pk, ok2 = transient(2 * T_FREE, ["D2", "D3", "mid_D23"], verbose, rec)
    out["D2_II2"] = d2 - pk["D2"][0]; out["D3_II2"] = d3 - pk["D3"][0]
    out["D2_end"] = -_disp("D2"); out["D1_x"] = _disp("D1", 1); out["D4_x"] = _disp("D4", 1)
    out["_ok"] = ok and ok2
    if with_pushdown and out["_ok"]:
        lf, hist = pushdown(area, verbose=verbose)
        out["LF_II"] = lf; out["_push"] = hist
    return out


def phase_III(verbose=True, rec=None):
    build(cut_D=True); add_test_load("test3"); gravity(verbose)
    d0 = _disp("B3")
    remove_column("B3", 0.0, verbose)
    pk, ok = transient(T_FREE, ["B3"], verbose, rec)
    return {"B3_III": d0 - pk["B3"][0], "_ok": ok}


BAND_D = (0.78, 1.12)
BAND_L = (0.82, 1.20)


def score(res, verbose=True):
    n_ok = 0
    rows = []
    for k, tv in TEST.items():
        if k not in res:
            continue
        r = res[k] / tv
        band = BAND_L if k.startswith("LF") else BAND_D
        ok = band[0] <= r <= band[1]
        n_ok += ok
        rows.append((k, res[k], tv, r, ok))
    if verbose:
        for k, m, tv, r, ok in rows:
            print(f"  {k:8s} model {m:7.2f}  test {tv:7.2f}  ratio {r:5.2f}  {'PASS' if ok else 'FAIL'}")
        print(f"  {n_ok}/{len(rows)} in band")
    return n_ok, rows


def demo():
    import json
    t0 = time.time()
    res, recs = {}, {}
    for name, ph in (("I", phase_I), ("II", phase_II), ("III", phase_III)):
        rec = []
        r = ph(verbose=True, rec=rec) if name == "III" else ph(verbose=True, rec=rec, with_pushdown=True)
        push = r.pop("_push", None)
        print(f"  phase {name}: " + "  ".join(f"{k} {v:.2f}" if isinstance(v, float) else f"{k} {v}" for k, v in r.items()),
              f"[{time.time()-t0:.0f} s]")
        res.update({k: v for k, v in r.items() if not k.startswith("_")})
        recs[name] = np.array(rec)
        if push is not None:
            recs["push_" + name] = np.array(push)
    n_ok, rows = score(res)
    wall = time.time() - t0
    json.dump(dict(res=res, rows=rows, n_ok=n_ok, n_tot=len(rows), wall_s=wall,
                   params=dict(EC_MULT=EC_MULT, FT_MULT=FT_MULT, ETS_DIV=ETS_DIV, T_RELEASE=T_RELEASE, T_FREE=T_FREE,
                               DT=DT, ZETA=ZETA, TRANSF=TRANSF, OFFSETS=OFFSETS, SHELL=SHELL, NXB=NXB, NYB=NYB)),
              open("xiao2015_calibration.json", "w"), indent=1)
    np.savez("xiao2015_histories.npz", **recs)
    print(f"  demo: {n_ok}/{len(rows)} in band, {wall:.0f} s")


if __name__ == "__main__":
    demo()
