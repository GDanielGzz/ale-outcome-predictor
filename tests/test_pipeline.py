"""Pipeline + module integration tests.

Replaces the scaffold's smoke-test TODO with real coverage:
* pure-Python core of ``baseline_model`` (splits + metrics) — no deps;
* ``feature_engineer`` GPR parsing — no solver;
* ``media`` exchange-bound resolution — needs a model;
* a fast end-to-end pass on the bundled ``textbook`` model (e_coli_core) with a
  tiny synthetic corpus: features -> labels -> train -> rank.

The heavy iJO1366 run is exercised by ``scripts/run_experiment.py``; these tests
stay offline + seconds-fast so CI stays green without an ALEdb pull.
"""

from __future__ import annotations

import pytest

# ----------------------------- pure-Python core ----------------------------- #


def test_cluster_aware_split_keeps_groups_whole():
    from ale_outcome_predictor.baseline_model import cluster_aware_split_indices

    groups = [f"g{i//4}" for i in range(40)]  # 10 clusters of 4
    splits = cluster_aware_split_indices(groups, n_splits=5, seed=0)
    assert len(splits) == 5
    for train, test in splits:
        train_groups = {groups[i] for i in train}
        test_groups = {groups[i] for i in test}
        assert train_groups.isdisjoint(test_groups)  # no cluster leaks across split


def test_top_k_recall_and_baseline():
    from ale_outcome_predictor.baseline_model import (
        top_k_recall,
        uniform_essential_baseline_scores,
    )

    y = [1, 0, 0, 1, 0]
    scores = [0.9, 0.1, 0.2, 0.8, 0.05]
    assert top_k_recall(y, scores, k=2) == 1.0          # both positives in top 2
    assert top_k_recall([0, 0], [0.1, 0.2], k=1) != top_k_recall([0, 0], [0.1, 0.2], k=1) \
        or True  # nan path doesn't raise
    base = uniform_essential_baseline_scores([True, False, None])
    assert base[0] >= 0.5 > base[1]                     # essential outranks non-essential


def test_year_split():
    from ale_outcome_predictor.baseline_model import extract_year, year_split_indices

    assert extract_year("Phaneuf 2019 NAR") == 2019
    tr, te = year_split_indices([2006, 2015, 2021, None], cutoff=2020)
    assert tr == [0, 1] and te == [2]                   # None placed on neither side


def test_gpr_complexity_parsing():
    from ale_outcome_predictor.feature_engineer import gpr_complexity

    c = gpr_complexity("b0001 and b0002 or b0003")
    assert c["n_genes"] == 3
    assert c["is_isozyme"] is True and c["is_in_complex"] is True
    assert gpr_complexity("")["n_genes"] == 0


def test_coding_effect_word_boundary():
    # the classic trap: 'nonsynonymous' must not be swallowed by 'synonymous'
    from ale_outcome_predictor.data_loader import _map_coding_effect

    assert _map_coding_effect("nonsynonymous (S450L)") == "missense"
    assert _map_coding_effect("synonymous") == "synonymous"


# ------------------------------- needs cobra -------------------------------- #


@pytest.fixture(scope="module")
def core_model():
    cobra = pytest.importorskip("cobra")
    return cobra.io.load_model("textbook")  # e_coli_core, bundled, fast


def test_media_resolves_carbon_swap(core_model):
    from ale_outcome_predictor.media import resolve_exchange_bounds

    b = resolve_exchange_bounds("glucose M9 minimal 37C", core_model)
    assert b.get("EX_glc__D_e", (0, 0))[0] < 0          # glucose uptake opened
    b2 = resolve_exchange_bounds("acetate minimal", core_model)
    if "EX_ac_e" in core_model.reactions:
        assert b2["EX_ac_e"][0] < 0                      # acetate opened
        assert b2.get("EX_glc__D_e", (0, 0))[0] == 0     # glucose closed


def test_end_to_end_textbook(core_model):
    """features -> labels -> train -> ranked output on e_coli_core."""
    pytest.importorskip("sklearn")
    import pandas as pd

    from ale_outcome_predictor.baseline_model import (
        ModelConfig,
        predict_ranked_genes,
        train_model,
    )
    from ale_outcome_predictor.feature_engineer import (
        FeatureConfig,
        attach_labels,
        build_feature_matrix,
        validate_features,
    )
    from ale_outcome_predictor.media import make_env_resolver

    conditions = ["glucose minimal 37C", "glucose minimal 42C thermal"]
    feats = build_feature_matrix(
        core_model,
        conditions=conditions,
        config=FeatureConfig(processes=1),
        env_resolver=make_env_resolver(core_model),
    )
    validate_features(feats)
    assert len(feats) == len(core_model.genes) * len(conditions)

    # synthetic corpus: pick two real model genes as "convergent" under one condition
    some_genes = [g.id for g in list(core_model.genes)[:2]]
    corpus = pd.DataFrame(
        {
            "selection_condition": ["glucose minimal 37C", "glucose minimal 37C"],
            "gene": some_genes,
        }
    )
    feats = attach_labels(feats, corpus, key="selection_condition")
    assert int(feats["mutated"].fillna(False).astype("boolean").sum()) == 2

    bundle = train_model(feats, ModelConfig(n_estimators=30, min_child_samples=1, num_leaves=7))
    ranked = predict_ranked_genes(bundle, feats[feats["selection_condition"] == "glucose minimal 37C"])
    assert list(ranked.columns) == ["gene", "selection_condition", "probability"]
    # the two planted positives should top the ranking for that condition
    top2 = set(ranked.head(2)["gene"])
    assert set(some_genes) == top2


# ----------------------- gene-name -> b-number mapping ---------------------- #


def test_gene_map_names_synonyms_intergenic():
    from ale_outcome_predictor.gene_map import load_gene_aliases, map_name

    al = load_gene_aliases()
    # the 10 corpus genes resolve to their known b-numbers
    assert map_name("pyrE", al) == "b3642"
    assert map_name("glpK", al) == "b3926"
    assert map_name("fabA", al) == "b0954"
    # synonyms
    assert map_name("icdA", al) == "b1136"          # icd synonym
    # primary wins over a synonym collision: fabG is b1093, not accC's synonym
    assert map_name("fabG", al) == "b1093"
    # intergenic / multi-gene loci resolve to the first mappable flank
    assert map_name("pyrE/rph", al) == "b3642"
    assert map_name("hns ← tdk", al) == "b1237"
    assert map_name("not_a_real_gene", al) is None


# --------------------- real ALEdb GLU integration --------------------------- #


def test_glu_converged_genes_map_and_in_corpus():
    import pandas as pd

    from ale_outcome_predictor.gene_map import load_gene_aliases, map_name

    al = load_gene_aliases()
    # a sample of the real ExpID762 converged genes -> verified b-numbers
    for name, b in {"pykF": "b1676", "glgC": "b3430", "putA": "b1014",
                    "xylB": "b3564", "fadE": "b0221", "corA": "b3816"}.items():
        assert map_name(name, al) == b
    # the curated corpus now carries the real converged glucose positives
    from pathlib import Path
    corpus = pd.read_csv(Path(__file__).parents[1] / "corpus" / "curated_ale_corpus.csv")
    glc = corpus[corpus["selection_condition"].astype(str).str.contains("glucose M9 minimal 37C")]
    assert len(glc) >= 15
    assert (glc["experiment_id"].astype(str).str.contains("ExpID762")).all()


# --------------------- real ALEdb thermal (ExpID740) ------------------------ #


def test_tenaillon_thermal_genes_map_and_in_corpus():
    import pandas as pd
    from pathlib import Path

    from ale_outcome_predictor.gene_map import load_gene_aliases, map_name

    al = load_gene_aliases()
    for name, b in {"murA": "b3189", "rbsA": "b3749", "glmS": "b3729",
                    "mrdA": "b0635", "fbaB": "b2097"}.items():
        assert map_name(name, al) == b
    corpus = pd.read_csv(Path(__file__).parents[1] / "corpus" / "curated_ale_corpus.csv")
    thermal = corpus[corpus["selection_condition"].astype(str).str.contains("42C thermal")]
    assert len(thermal) >= 40
    assert (thermal["experiment_id"].astype(str).str.contains("ExpID740")).all()


# --------------------- real ALEdb glycerol (ExpID1523) ---------------------- #


def test_glycerol_glpk_real_converged():
    import pandas as pd
    from pathlib import Path

    corpus = pd.read_csv(Path(__file__).parents[1] / "corpus" / "curated_ale_corpus.csv")
    gly = corpus[corpus["selection_condition"].astype(str).str.contains("glycerol")]
    glpk = gly[gly["gene_name"] == "glpK"].iloc[0]
    assert glpk["gene"] == "b3926"
    assert "ExpID1523" in str(glpk["experiment_id"])     # now real ALEdb-confirmed


# --------------------- benzoate / aromatic stress axis ---------------------- #


def test_aromatic_stress_axis():
    from ale_outcome_predictor.feature_engineer import STRESS_CLASSES, parse_selection_condition

    assert "aromatic" in STRESS_CLASSES                       # new axis wired
    assert parse_selection_condition("M9 glucose benzoate tolerization 37C").stress_class == "aromatic"
    assert parse_selection_condition("M9 glucose ferulic acid 37C").stress_class == "aromatic"
    assert parse_selection_condition("M9 glucose p-coumaric 37C").stress_class == "aromatic"
    # generic acid still maps to pH (aromatic-acid keywords are checked first)
    assert parse_selection_condition("M9 glucose low pH acid").stress_class == "ph"


# ----------------- benzoate + acetate real ALEdb selections ----------------- #


def test_benzoate_acetate_genes_and_corpus():
    import pandas as pd
    from pathlib import Path

    from ale_outcome_predictor.gene_map import load_gene_aliases, map_name

    al = load_gene_aliases()
    for name, b in {"folD": "b0529", "narH": "b1225", "tdcD": "b3115",   # benzoate
                    "chbC": "b1737", "dacD": "b2010"}.items():           # acetate
        assert map_name(name, al) == b
    corpus = pd.read_csv(Path(__file__).parents[1] / "corpus" / "curated_ale_corpus.csv")
    conds = set(corpus["selection_condition"].astype(str))
    assert any("benzoate" in c for c in conds)
    assert any("acetate" in c for c in conds)
    assert len(corpus) >= 80                                            # 86 positives, 5 conditions


# -------------------- sequence-feature fusion (milestone 16) ----------------- #


def test_sequence_features_biophysical_and_merge(tmp_path):
    pytest.importorskip("Bio")
    import pandas as pd

    from ale_outcome_predictor.sequence_features import (
        BIOPHYS_COLUMNS, biophysical_features, build_biophysical_table,
        load_proteome_fasta, merge_sequence_features,
    )

    feats = biophysical_features("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDG")
    assert set(BIOPHYS_COLUMNS) <= set(feats) and feats["seq_length"] == 43

    fa = tmp_path / "syn.fasta"
    fa.write_text(">sp|P0A7E3|X OS=E. coli GN=pyrE PE=1\nMKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ\n"
                  ">lcl|x [locus_tag=b3926] glpK\nMTEKKYIVALDQGTTSSRAVVMDHDANIISVSQ\n")
    prot = load_proteome_fasta(str(fa))
    tbl = build_biophysical_table(["b3642", "b3926", "b9999"], prot)   # pyrE via GN, glpK direct, absent
    assert not pd.isna(tbl.loc["b3642", "seq_pi"])      # resolved gene-name -> b-number
    assert not pd.isna(tbl.loc["b3926", "seq_pi"])
    assert pd.isna(tbl.loc["b9999", "seq_pi"])          # no sequence -> NA
    fm = pd.DataFrame({"gene": ["b3642"], "selection_condition": ["x"]})
    assert "seq_pi" in merge_sequence_features(fm, tbl).columns
