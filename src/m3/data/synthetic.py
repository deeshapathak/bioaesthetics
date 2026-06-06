"""Synthetic data generator with a planted old→young signal (synthetic mode).

This is *not* the science — it is a test fixture that lets the whole pipeline
run end-to-end with no external accounts, while exercising real signal:

  * a true aging direction (some genes up in old, some down),
  * GTEx-like skin at two sites + external aging cohorts (for Signature A + meta),
  * a photoexposure/site axis (Signature B),
  * curated programs (Signature C: senescence/SASP, ECM/collagen, inflammation),
  * an L1000-like compound library where *known actives* reverse the signature,
    *aging-promoters* worsen it (directional controls), a few *novel* compounds
    also reverse it (the hypotheses), and the rest are noise.

Everything is deterministic given the config seed. The generator also returns
the ground truth, which the validation stage (A4) uses for blind recovery —
but which the scoring stages never see.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd


@dataclasses.dataclass
class Cohort:
    name: str
    expr: pd.DataFrame      # samples x genes
    meta: pd.DataFrame      # samples x covariates (group, sex, RIN, ischemic_time, age, site)


@dataclasses.dataclass
class SyntheticData:
    gene_ids: list[str]
    landmark_genes: list[str]
    gtex_sites: dict[str, Cohort]          # site name -> cohort (Signatures A & B)
    external_cohorts: list[Cohort]         # additional aging series (A2.1 meta)
    gene_sets: dict[str, list[str]]        # Signature C curated programs
    l1000_signatures: np.ndarray           # (n_sigs, n_landmark) moderated z-scores
    l1000_meta: pd.DataFrame               # per-signature: compound, cell_line, dose
    compound_meta: pd.DataFrame            # per-compound: label, moa
    truth: dict                            # ground truth (validation only)


# Recognizable names so face validity (A4) is meaningful in synthetic mode.
_KNOWN_ACTIVE_NAMES = [
    "tretinoin", "retinol", "adapalene", "tazarotene",        # retinoids
    "rapamycin", "everolimus", "metformin",                   # geroprotectors
    "dasatinib", "quercetin", "fisetin", "navitoclax",        # senolytics
    "spermidine", "resveratrol", "NAD-precursor",
]
_PROMOTER_NAMES = [
    "doxorubicin", "etoposide", "bleomycin", "cisplatin",     # genotoxic / senescence-inducing
    "hydrogen-peroxide", "paraquat", "UVB-mimetic",
]


def generate(cfg) -> SyntheticData:
    s = cfg.get("synthetic")
    rng = np.random.default_rng(cfg.seed)

    n_genes = int(s["n_genes"])
    n_landmark = int(s["n_landmark"])
    n_aging = int(s["n_aging_genes"])
    eff = s["effect"]

    gene_ids = [f"G{i:05d}" for i in range(n_genes)]
    gidx = {g: i for i, g in enumerate(gene_ids)}

    # ---- landmark (L1000-measured) gene subset ------------------------------
    landmark_idx = np.sort(rng.choice(n_genes, size=n_landmark, replace=False))
    landmark_genes = [gene_ids[i] for i in landmark_idx]

    # ---- true aging direction `a` (half up, half down in old) ---------------
    aging_idx = rng.choice(n_genes, size=n_aging, replace=False)
    a = np.zeros(n_genes)
    half = n_aging // 2
    a[aging_idx[:half]] = +eff["age_beta"]          # higher in old
    a[aging_idx[half:]] = -eff["age_beta"]          # lower in old
    up_genes = [gene_ids[i] for i in aging_idx[:half]]
    down_genes = [gene_ids[i] for i in aging_idx[half:]]

    # ---- photoexposure / site direction `svec` (Signature B) ----------------
    site_idx = rng.choice(n_genes, size=n_aging // 2, replace=False)
    svec = np.zeros(n_genes)
    svec[site_idx] = rng.choice([-1.0, 1.0], size=site_idx.size) * eff["site_beta"]

    # ---- nuisance covariate gene loadings (so covarying them matters) -------
    load_sex = rng.normal(0, 0.6, n_genes)
    load_rin = rng.normal(0, 0.4, n_genes)
    load_isch = rng.normal(0, 0.4, n_genes)
    baseline = rng.normal(6.0, 1.0, n_genes)
    noise_sd = float(eff["noise"])

    def make_cohort(name, n_donors, site_value, age_dir=1.0, batch=0.0):
        """One expression cohort with the planted age effect."""
        n_young = n_donors // 2
        n_old = n_donors - n_young
        age = np.concatenate([rng.uniform(22, 38, n_young), rng.uniform(55, 72, n_old)])
        group = (age >= 50).astype(int)              # 1 = old
        age_c = (age - age.mean()) / age.std()
        sex = rng.integers(0, 2, n_donors).astype(float)
        rin = rng.uniform(5.5, 9.5, n_donors)
        isch = rng.uniform(0, 1500, n_donors)
        rin_c = (rin - rin.mean()) / rin.std()
        isch_c = (isch - isch.mean()) / isch.std()

        X = (
            baseline[None, :]
            + age_dir * np.outer(age_c, a)                 # aging effect
            + site_value * svec[None, :]                   # site/photoexposure
            + np.outer(sex, load_sex)
            + np.outer(rin_c, load_rin)
            + np.outer(isch_c, load_isch)
            + batch                                        # study batch offset
            + rng.normal(0, noise_sd, (n_donors, n_genes))
        )
        expr = pd.DataFrame(X, columns=gene_ids,
                            index=[f"{name}_S{i:03d}" for i in range(n_donors)])
        meta = pd.DataFrame(
            {"group": group, "sex": sex, "RIN": rin, "ischemic_time": isch,
             "age": age, "site": name},
            index=expr.index,
        )
        return Cohort(name=name, expr=expr, meta=meta)

    # ---- GTEx: two skin sites (suprapubic = non-sun, lower leg = sun) --------
    gtex_sites = {
        "suprapubic": make_cohort("suprapubic", int(s["gtex_donors"]), site_value=0.0),
        "lower_leg": make_cohort("lower_leg", int(s["gtex_donors"]), site_value=1.0),
    }

    # ---- external aging cohorts (A2.1 meta) ---------------------------------
    external_cohorts = [
        make_cohort("ext_skin_cohort", int(s["external_cohort_donors"]),
                    site_value=0.0, batch=rng.normal(0, 0.3, n_genes)),
        make_cohort("tabula_muris_senis", int(s["external_cohort_donors"]),
                    site_value=0.0, age_dir=0.85, batch=rng.normal(0, 0.5, n_genes)),
    ]

    # ---- curated programs (Signature C) -------------------------------------
    def program(seed_genes, extra_pool, n_extra):
        pool = [g for g in gene_ids if g not in seed_genes]
        extra = rng.choice(pool, size=min(n_extra, len(pool)), replace=False).tolist()
        return list(dict.fromkeys(list(seed_genes) + extra))

    gene_sets = {
        # senescence/SASP & inflammation skew toward genes UP in old
        "SenMayo": program(up_genes[:40], up_genes, 30),
        "MSigDB_inflammation": program(up_genes[20:55], up_genes, 25),
        # ECM / collagen / elastin skew toward genes DOWN in old
        "ECM_collagen": program(down_genes[:40], down_genes, 30),
        "CellAge": program(up_genes[:25] + down_genes[:25], up_genes, 20),
    }

    # ---- L1000-like compound library ----------------------------------------
    a_landmark = a[landmark_idx]                      # aging direction in L1000 space
    a_landmark_n = a_landmark / (np.linalg.norm(a_landmark) + 1e-9)

    cell_lines = list(s["cell_lines"])
    doses = list(s["doses"])
    skin_relevant = set(cfg.get("connectivity.skin_relevant_cell_lines", []))

    n_comp = int(s["n_compounds"])
    n_known = int(s["n_known_actives"])
    n_prom = int(s["n_aging_promoters"])
    n_novel = 8

    labels = (["known_active"] * n_known + ["aging_promoter"] * n_prom +
              ["novel"] * n_novel)
    labels += ["random"] * (n_comp - len(labels))
    rng.shuffle(labels)

    names, label_col, moa_col = [], [], []
    ka_i = pr_i = nv_i = rn_i = 0
    for lab in labels:
        if lab == "known_active":
            nm = _KNOWN_ACTIVE_NAMES[ka_i % len(_KNOWN_ACTIVE_NAMES)] + (
                "" if ka_i < len(_KNOWN_ACTIVE_NAMES) else f"_{ka_i}")
            moa = "known dermatology/longevity intervention"; ka_i += 1
        elif lab == "aging_promoter":
            nm = _PROMOTER_NAMES[pr_i % len(_PROMOTER_NAMES)] + (
                "" if pr_i < len(_PROMOTER_NAMES) else f"_{pr_i}")
            moa = "senescence-inducing / genotoxic"; pr_i += 1
        elif lab == "novel":
            nm = f"BIO-{nv_i:03d}"; moa = "uncharacterized"; nv_i += 1
        else:
            nm = f"cpd-{rn_i:04d}"; moa = "unknown"; rn_i += 1
        names.append(nm); label_col.append(lab); moa_col.append(moa)

    compound_meta = pd.DataFrame({"label": label_col, "moa": moa_col}, index=names)
    # de-duplicate any accidental name clashes
    compound_meta = compound_meta[~compound_meta.index.duplicated(keep="first")]
    names = list(compound_meta.index)

    rev_strength = float(eff["reverser_strength"])
    rows, sig_list = [], []
    for nm in names:
        lab = compound_meta.loc[nm, "label"]
        if lab in ("known_active", "novel"):
            base = -rev_strength * a_landmark_n        # reverse aging
        elif lab == "aging_promoter":
            base = +rev_strength * a_landmark_n        # worsen aging
        else:
            base = np.zeros(n_landmark)                # random: no directional signal
        for cl in cell_lines:
            # skin-relevant lines show the effect more strongly; others attenuate
            cl_gain = 1.0 if cl in skin_relevant else 0.55
            for d in doses:
                dose_gain = d / (d + 1.0)              # saturating dose-response
                z = (base * cl_gain * dose_gain * 12.0
                     + rng.normal(0, 1.0, n_landmark))  # moderated-z scale
                sig_list.append(z.astype(np.float32))
                rows.append((nm, cl, float(d)))

    l1000_meta = pd.DataFrame(rows, columns=["compound", "cell_line", "dose"])
    l1000_signatures = np.vstack(sig_list)

    truth = {
        "up_genes": up_genes,
        "down_genes": down_genes,
        "known_actives": [n for n in names if compound_meta.loc[n, "label"] == "known_active"],
        "aging_promoters": [n for n in names if compound_meta.loc[n, "label"] == "aging_promoter"],
        "novel": [n for n in names if compound_meta.loc[n, "label"] == "novel"],
        "n_aging_genes": int(n_aging),
    }

    return SyntheticData(
        gene_ids=gene_ids,
        landmark_genes=landmark_genes,
        gtex_sites=gtex_sites,
        external_cohorts=external_cohorts,
        gene_sets=gene_sets,
        l1000_signatures=l1000_signatures,
        l1000_meta=l1000_meta,
        compound_meta=compound_meta,
        truth=truth,
    )
