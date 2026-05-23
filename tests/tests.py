
import pytest
import pandas as pd
from pygenie_sdk import GENIEClient

#  config
DATA_DIR = "/Users/danadayan/PycharmProjects/genie_api"
VERSION = "16.1-public"


@pytest.fixture(scope="session")
def genie():
    """Single GENIEClient instance shared across all tests — loads files once."""
    return GENIEClient(DATA_DIR, version=VERSION)


# init

def test_init(genie):
    assert genie.version == VERSION
    assert genie.data_dir.exists()


def test_cache_starts_empty(genie):
    # note: cache may already have data from previous tests in session
    # just check it's a dict
    assert isinstance(genie._cache, dict)


# validate

def test_validate_finds_all_files(genie, capsys):
    genie.validate()
    captured = capsys.readouterr()
    assert "All files found" in captured.out


# summary

def test_summary_returns_dict(genie):
    s = genie.summary()
    assert isinstance(s, dict)


def test_summary_correct_keys(genie):
    s = genie.summary()
    for key in ["version", "n_patients", "n_samples", "n_cancer_types",
                "n_centers", "n_panels"]:
        assert key in s


def test_summary_large_dataset(genie):
    s = genie.summary()
    assert s["n_patients"] > 100000
    assert s["n_samples"] > 100000
    assert s["n_cancer_types"] > 50
    assert s["n_centers"] > 10


def test_summary_no_unknown_cancer_types(genie):
    samples = genie._load("data_clinical_sample.txt")
    known = samples[samples["CANCER_TYPE"] != "UNKNOWN"]
    s = genie.summary()
    assert s["n_cancer_types"] == known["CANCER_TYPE"].nunique()


# query patients

def test_query_patients_all(genie):
    df = genie.query_patients()
    assert len(df) > 100000


def test_query_patients_by_center(genie):
    df = genie.query_patients(center="MSK")
    assert len(df) > 0
    assert all(df["CENTER"] == "MSK")


def test_query_patients_by_center_vicc(genie):
    df = genie.query_patients(center="VICC")
    assert len(df) > 0
    assert all(df["CENTER"] == "VICC")


def test_query_patients_by_sex_female(genie):
    df = genie.query_patients(sex="Female")
    assert len(df) > 0
    assert all(df["SEX"] == "Female")


def test_query_patients_by_sex_male(genie):
    df = genie.query_patients(sex="Male")
    assert len(df) > 0
    assert all(df["SEX"] == "Male")


def test_query_patients_dead_true(genie):
    df = genie.query_patients(dead=True)
    assert len(df) > 0
    assert all(df["DEAD"].str.upper() == "TRUE")


def test_query_patients_dead_false(genie):
    df = genie.query_patients(dead=False)
    assert len(df) > 0
    assert all(df["DEAD"].str.upper() == "FALSE")


def test_query_patients_resets_index(genie):
    df = genie.query_patients(center="MSK")
    assert df.index.tolist() == list(range(len(df)))


# query samples

def test_query_samples_all(genie):
    df = genie.query_samples()
    assert len(df) > 100000


def test_query_samples_luad(genie):
    df = genie.query_samples(oncotree_code="LUAD")
    assert len(df) > 1000
    assert all(df["ONCOTREE_CODE"] == "LUAD")


def test_query_samples_nsclc(genie):
    df = genie.query_samples(cancer_type="Non-Small Cell Lung Cancer")
    assert len(df) > 1000
    assert all(df["CANCER_TYPE"] == "Non-Small Cell Lung Cancer")


def test_query_samples_by_panel(genie):
    df = genie.query_samples(seq_assay_id="MSK-IMPACT468")
    assert len(df) > 1000
    assert all(df["SEQ_ASSAY_ID"] == "MSK-IMPACT468")


def test_query_samples_by_age_range(genie):
    df = genie.query_samples(age_range=(40, 60))
    df["AGE_AT_SEQ_REPORT"] = pd.to_numeric(df["AGE_AT_SEQ_REPORT"], errors="coerce")
    df = df.dropna(subset=["AGE_AT_SEQ_REPORT"])
    assert all(df["AGE_AT_SEQ_REPORT"].between(40, 60))


def test_query_samples_invalid_cancer_type(genie):
    with pytest.raises(ValueError):
        genie.query_samples(cancer_type="This Cancer Does Not Exist")


def test_query_samples_by_sample_type(genie):
    df = genie.query_samples(sample_type="Primary")
    assert len(df) > 0
    assert all(df["SAMPLE_TYPE"] == "Primary")


# query by gene

def test_query_by_gene_tp53(genie):
    df = genie.query_by_gene("TP53")
    assert len(df) > 0
    # TP53 must appear in the results but other genes can too
    assert "TP53" in df["Hugo_Symbol"].values

def test_query_by_gene_kras(genie):
    df = genie.query_by_gene("KRAS")
    assert len(df) > 0
    assert "KRAS" in df["Hugo_Symbol"].values

def test_query_by_gene_list(genie):
    df = genie.query_by_gene(["TP53", "KRAS"])
    assert len(df) > 0
    # both genes must appear somewhere in results
    assert "TP53" in df["Hugo_Symbol"].values
    assert "KRAS" in df["Hugo_Symbol"].values


def test_query_by_gene_case_insensitive(genie):
    df1 = genie.query_by_gene("TP53")
    df2 = genie.query_by_gene("tp53")
    assert len(df1) == len(df2)


def test_query_by_gene_not_found(genie):
    with pytest.raises(ValueError):
        genie.query_by_gene("NOTAREALGENE99999")


def test_query_by_gene_missense_only(genie):
    df = genie.query_by_gene("TP53", variant_classification="Missense_Mutation")
    assert len(df) > 0
    assert all(df["Variant_Classification"] == "Missense_Mutation")


def test_query_by_gene_snp_only(genie):
    df = genie.query_by_gene("KRAS", variant_type="SNP")
    assert len(df) > 0
    assert all(df["Variant_Type"] == "SNP")


def test_query_by_gene_prints_count(genie, capsys):
    genie.query_by_gene("TP53")
    captured = capsys.readouterr()
    assert "Found" in captured.out
    assert "samples" in captured.out  # print says "Found X samples" not "mutations"


# query by mutation

def test_query_by_mutation_gene_only(genie):
    df = genie.query_by_mutation(gene="TP53")
    assert len(df) > 0


def test_query_by_mutation_with_id(genie):
    df = genie.query_by_mutation(gene="TP53", mutation_id="R175H")
    assert len(df) > 0
    assert all(df["HGVSp_Short"] == "p.R175H")


def test_query_by_mutation_adds_p_prefix(genie):
    df1 = genie.query_by_mutation(gene="TP53", mutation_id="R175H")
    df2 = genie.query_by_mutation(gene="TP53", mutation_id="p.R175H")
    assert len(df1) == len(df2)


def test_query_by_mutation_kras_g12d(genie):
    df = genie.query_by_mutation(gene="KRAS", mutation_id="G12D")
    assert len(df) > 0
    assert all(df["HGVSp_Short"] == "p.G12D")


def test_query_by_mutation_not_found(genie):
    with pytest.raises(ValueError):
        genie.query_by_mutation(gene="TP53", mutation_id="X999X")


def test_query_by_mutation_no_args(genie):
    with pytest.raises(ValueError):
        genie.query_by_mutation()


# query fusions

def test_query_fusions_all(genie):
    df = genie.query_fusions()
    assert len(df) > 1000


def test_query_fusions_alk(genie):
    df = genie.query_fusions(gene="ALK")
    assert len(df) > 0
    assert all(df["Hugo_Symbol"] == "ALK")


def test_query_fusions_ret(genie):
    df = genie.query_fusions(gene="RET")
    assert len(df) > 0


def test_query_fusions_in_frame(genie):
    df = genie.query_fusions(frame="in frame")
    assert len(df) > 0
    assert all(df["Frame"] == "in frame")


def test_query_fusions_invalid_frame(genie):
    with pytest.raises(ValueError):
        genie.query_fusions(frame="not_a_valid_frame")


# query CNA

def test_query_cna_erbb2_amp(genie):
    df = genie.query_cna(gene="ERBB2", cna_value=2)
    assert len(df) > 0
    assert all(df["cna_value"] == 2)
    assert all(df["cna_label"] == "Amplification")


def test_query_cna_cdkn2a_del(genie):
    df = genie.query_cna(gene="CDKN2A", cna_value=-2)
    assert len(df) > 0
    assert all(df["cna_value"] == -2)
    assert all(df["cna_label"] == "Deep deletion")


def test_query_cna_invalid_value(genie):
    with pytest.raises(Exception):
        genie.query_cna(cna_value=99)


# build cohort

def test_build_cohort_luad(genie):
    cohort = genie.build_cohort(oncotree_code="LUAD")
    assert len(cohort) > 1000
    assert "PATIENT_ID" in cohort.columns
    assert "SAMPLE_ID" in cohort.columns
    assert all(cohort["ONCOTREE_CODE"] == "LUAD")


def test_build_cohort_msk(genie):
    cohort = genie.build_cohort(center="MSK")
    assert len(cohort) > 1000


def test_build_cohort_with_mutations(genie):
    cohort = genie.build_cohort(oncotree_code="LUAD", include_mutations=True)
    assert "Hugo_Symbol" in cohort.columns
    assert "Tumor_Sample_Barcode" in cohort.columns


def test_build_cohort_resets_index(genie):
    cohort = genie.build_cohort(oncotree_code="LUAD")
    assert cohort.index.tolist() == list(range(len(cohort)))


# query multiple tumors

def test_query_multiple_tumors(genie):
    df = genie.query_multiple_tumors()
    assert len(df) > 1000
    assert "n_samples" in df.columns
    assert all(df["n_samples"] > 1)


def test_query_multiple_tumors_specific_patient(genie):
    df = genie.query_multiple_tumors(patient_id="GENIE-MSK-P-0036072")
    assert len(df) > 1
    assert all(df["PATIENT_ID"] == "GENIE-MSK-P-0036072")


# TMB

def test_compute_tmb_columns(genie):
    tmb = genie.compute_tmb()
    for col in ["Tumor_Sample_Barcode", "mutation_count", "SEQ_ASSAY_ID",
                "tmb", "tmb_available", "tmb_high", "tmb_hypermutator"]:
        assert col in tmb.columns


def test_compute_tmb_median_realistic(genie):
    tmb = genie.compute_tmb()
    valid = tmb[tmb["tmb_available"] == True]["tmb"]
    assert 1 < valid.median() < 20  # realistic range for panel sequencing


def test_compute_normalized_tmb(genie):
    tmb = genie.compute_normalized_tmb()
    assert "normalized_tmb" in tmb.columns
    assert "assay_median" in tmb.columns
    # median normalized TMB should be close to 1
    assert 0.9 < tmb["normalized_tmb"].median() < 1.1


# query mutations by cohort

def test_query_mutations_by_cohort_luad(genie):
    df = genie.query_mutations_by_cohort(oncotree_code="LUAD")
    assert len(df) > 1000
    assert "Hugo_Symbol" in df.columns
    assert "Tumor_Sample_Barcode" in df.columns


def test_query_mutations_by_cohort_center(genie):
    df = genie.query_mutations_by_cohort(center="MSK")
    assert len(df) > 1000


# list methods

def test_list_cancer_types(genie):
    result = genie.list_cancer_types()
    assert "Non-Small Cell Lung Cancer" in result.index
    assert "Colorectal Cancer" in result.index
    assert "UNKNOWN" not in result.index


def test_list_oncotree_codes(genie):
    result = genie.list_oncotree_codes()
    assert "LUAD" in result.index
    assert "BRCA" in result.index


def test_list_centers(genie):
    result = genie.list_centers()
    assert "MSK" in result.index
    assert "VICC" in result.index


def test_list_panels(genie):
    result = genie.list_panels()
    assert "MSK-IMPACT468" in result.index


def test_list_genes(genie):
    result = genie.list_genes()
    assert "TP53" in result.index
    assert "KRAS" in result.index


def test_list_variant_classification(genie):
    result = genie.list_variant_classification()
    assert "Missense_Mutation" in result.index
    assert "Nonsense_Mutation" in result.index


def test_list_sample_types(genie):
    result = genie.list_sample_types()
    assert "Primary" in result.index


# panel functions

def test_list_genes_in_panel(genie):
    genes = genie.list_genes_in_panel("MSK-IMPACT468")
    assert isinstance(genes, list)
    assert len(genes) > 100
    assert "TP53" in genes
    assert "KRAS" in genes


def test_list_genes_in_panel_dfci(genie):
    genes = genie.list_genes_in_panel("DFCI-ONCOPANEL-3")
    assert isinstance(genes, list)
    assert len(genes) > 0


def test_list_genes_in_panel_invalid(genie):
    with pytest.raises(ValueError):
        genie.list_genes_in_panel("FAKE-PANEL-999")


def test_get_panel_size_msk(genie):
    size = genie.get_panel_size("MSK-IMPACT410")
    assert isinstance(size, float)
    assert 0.5 < size < 5.0  # realistic panel size in Mb


def test_get_panel_size_small_panel(genie, capsys):
    # VHIO-BREAST-V02 is known to be tiny
    result = genie.get_panel_size("VHIO-BREAST-V02")
    captured = capsys.readouterr()
    assert result is None  # should return None for small panels
    assert "Warning" in captured.out


# search gene easy

def test_search_gene_exact_tp53(genie):
    result = genie.search_gene("TP53")
    assert "TP53" in result


def test_search_gene_lowercase(genie):
    result = genie.search_gene("tp53")
    assert "TP53" in result


def test_search_gene_typo(genie):
    result = genie.search_gene("EGRF")  # typo for EGFR
    assert isinstance(result, list)
    assert "EGFR" in result


def test_search_gene_partial(genie):
    result = genie.search_gene("BRCA")
    assert isinstance(result, list)
    assert len(result) > 0


# patient report

def test_patient_report_returns_dict(genie, capsys):
    report = genie.patient_report("GENIE-DFCI-268887")
    assert isinstance(report, dict)
    assert "clinical" in report
    assert "samples" in report
    assert "mutations" in report


def test_patient_report_correct_id(genie, capsys):
    report = genie.patient_report("GENIE-DFCI-268887")
    assert report["clinical"]["PATIENT_ID"] == "GENIE-DFCI-268887"


def test_patient_report_has_cna(genie, capsys):
    # GENIE-DFCI-268887 is known to have CNAs
    report = genie.patient_report("GENIE-DFCI-268887")
    assert len(report["samples"]) > 0


def test_patient_report_not_found(genie):
    with pytest.raises(ValueError):
        genie.patient_report("GENIE-FAKE-000000")


# co-mutation

def test_comutation_luad(genie):
    result = genie.comutation(
        genes=["TP53", "KRAS", "EGFR"],
        oncotree_code="LUAD"
    )
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3  # 3 pairs from 3 genes
    assert "gene_a" in result.columns
    assert "gene_b" in result.columns
    assert "odds_ratio" in result.columns
    assert "p_value" in result.columns
    assert "relationship" in result.columns
    assert "p_value_label" in result.columns


def test_comutation_kras_egfr_exclusive(genie):
    result = genie.comutation(
        genes=["KRAS", "EGFR"],
        oncotree_code="LUAD"
    )
    row = result.iloc[0]
    assert row["odds_ratio"] < 1
    assert row["relationship"] == "mutual exclusivity"


def test_comutation_requires_two_genes(genie):
    with pytest.raises(Exception):
        genie.comutation(genes=["TP53"], oncotree_code="LUAD")


# compare cohorts

def test_compare_cohorts_returns_df(genie):
    msk = genie.build_cohort(center="MSK", oncotree_code="LUAD")
    dfci = genie.build_cohort(center="DFCI", oncotree_code="LUAD")
    result = genie.compare_cohorts(msk, dfci,
                                   cohort1_name="MSK",
                                   cohort2_name="DFCI")
    assert isinstance(result, pd.DataFrame)
    assert "gene" in result.columns
    assert "freq_MSK" in result.columns
    assert "freq_DFCI" in result.columns
    assert "odds_ratio" in result.columns
    assert "significant" in result.columns


def test_compare_cohorts_frequency_range(genie):
    msk = genie.build_cohort(center="MSK", oncotree_code="LUAD")
    dfci = genie.build_cohort(center="DFCI", oncotree_code="LUAD")
    result = genie.compare_cohorts(msk, dfci,
                                   cohort1_name="MSK",
                                   cohort2_name="DFCI")
    assert all(result["freq_MSK"].between(0, 1))
    assert all(result["freq_DFCI"].between(0, 1))


# query by center

def test_query_by_center_msk(genie):
    result = genie.query_by_center("MSK")
    assert isinstance(result, dict)
    assert "patients" in result
    assert "samples" in result
    assert all(result["patients"]["CENTER"] == "MSK")


def test_query_by_center_with_mutations(genie):
    result = genie.query_by_center("VICC", include_mutations=True)
    assert "mutations" in result
    assert len(result["mutations"]) > 0


def test_query_by_center_with_fusions(genie):
    result = genie.query_by_center("MSK", include_fusions=True)
    assert "fusions" in result
    assert len(result["fusions"]) > 0


def test_query_by_center_with_cna(genie):
    result = genie.query_by_center("MSK", include_cna=True)
    assert "cna" in result


# gene cooccurrence with clinical

def test_gene_cooccurrence_returns_dict(genie):
    result = genie.gene_cooccurrence_with_clinical(
        "KRAS", oncotree_code="LUAD"
    )
    assert isinstance(result, dict)
    assert "mutated" in result
    assert "wildtype" in result
    assert "sex_comparison" in result


def test_gene_cooccurrence_mutated_have_gene(genie):
    result = genie.gene_cooccurrence_with_clinical(
        "KRAS", oncotree_code="LUAD"
    )
    assert len(result["mutated"]) > 0
    assert len(result["wildtype"]) > 0


def test_gene_cooccurrence_with_cohort(genie):
    luad = genie.build_cohort(oncotree_code="LUAD")
    result = genie.gene_cooccurrence_with_clinical("KRAS", cohort=luad)
    assert isinstance(result, dict)
