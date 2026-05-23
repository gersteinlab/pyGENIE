import pandas as pd
from pathlib import Path

class GENIEClient:

    def __init__(self, data_dir, version=None):
        self.data_dir = Path(data_dir)
        self.version = version
        self._cache = {}

    def _load(self, filename):
        parquet_path = self.data_dir / (filename + ".parquet")

        # if parquet exists, load it — much faster than TSV
        if parquet_path.exists():
            return pd.read_parquet(parquet_path)

        # otherwise load TSV and save parquet for next time
        df = pd.read_csv(self.data_dir / filename, sep="\t",
                         comment="#", low_memory=False)
        df.to_parquet(parquet_path, index=False)
        return df

    def clear_cache(self):
        self._cache.clear()
        print("Cache cleared.")

    def validate(self):
        files = {
            "data_clinical_patient.txt": "Patient clinical data",
            "data_clinical_sample.txt": "Sample clinical data",
            "data_mutations_extended.txt": "Mutations",
            "data_fusions.txt": "Fusions",
            "data_CNA.txt": "Copy number alterations",
            "genomic_information.txt": "Genomic panel information",
            "data_gene_matrix.txt": "Gene matrix",
        }

        print(f"\n----- GENIE SDK Validation -----")
        print(f"    data_dir: {self.data_dir}")
        print(f"    version: {self.version}")

        all_good = True
        for filename, description in files.items():
            filepath = self.data_dir / filename
            if filepath.exists():
                print(f" v {description} ({filename})")
            else:
                print(f" x {description} ({filename})")
                all_good = False
        print()
        if all_good:
            print("     All files found")
        else:
            print("     Some files not found. Download from Synapse")

    def summary(self):
        patients = self._load("data_clinical_patient.txt")
        samples = self._load("data_clinical_sample.txt")

        known_cancers = samples[samples["CANCER_TYPE"] != "UNKNOWN"]

        return {
            "version": self.version,
            "n_patients": len(patients),
            "n_samples": len(samples),
            "n_cancer_types": known_cancers["CANCER_TYPE"].nunique(),
            "n_centers": patients["CENTER"].nunique(),
            "n_panels": samples["SEQ_ASSAY_ID"].nunique(),
        }

    def query_patients(self, center=None, sex=None, dead=None):
        df = self._load("data_clinical_patient.txt")

        if center:
            df = df[df["CENTER"] == center]

        if sex:
            df = df[df["SEX"] == sex]

        if dead is not None:
            if dead == True:
                df = df[df["DEAD"].str.upper() == "TRUE"]
            elif dead == False:
                df = df[df["DEAD"].str.upper() == "FALSE"]
            else:
                df = df[df["DEAD"] == dead]

        return df.reset_index(drop=True)

    def query_samples(self, cancer_type=None, oncotree_code=None, age_range=None,
                      seq_assay_id=None, sample_type=None):
        df = self._load("data_clinical_sample.txt")

        if cancer_type:
            if cancer_type not in df["CANCER_TYPE"].values:
                raise ValueError(f"'{cancer_type}' not found. Use list_cancer_types() to see valid options.")
            df = df[df["CANCER_TYPE"] == cancer_type]

        if oncotree_code:
            df = df[df["ONCOTREE_CODE"] == oncotree_code]

        if seq_assay_id:
            df = df[df["SEQ_ASSAY_ID"] == seq_assay_id]

        if sample_type:
            df = df[df["SAMPLE_TYPE"] == sample_type]

        if age_range:
            df["AGE_AT_SEQ_REPORT"] = pd.to_numeric(df["AGE_AT_SEQ_REPORT"], errors="coerce")
            df = df[df["AGE_AT_SEQ_REPORT"].between(age_range[0], age_range[1])]

        return df.reset_index(drop=True)

    def list_cancer_types(self):
        df = self._load("data_clinical_sample.txt")
        counts = df["CANCER_TYPE"].value_counts()
        counts = counts[counts.index != "UNKNOWN"]
        return counts

    def list_oncotree_codes(self):
        df = self._load("data_clinical_sample.txt")
        counts = df["ONCOTREE_CODE"].value_counts()
        counts = counts[counts.index != "UNKNOWN"]
        return counts

    def list_centers(self):
        df = self._load("data_clinical_patient.txt")
        counts = df["CENTER"].value_counts()
        return counts

    def list_panels(self):
        df = self._load("data_clinical_sample.txt")
        counts = df["SEQ_ASSAY_ID"].value_counts()
        return counts

    def list_genes(self):
        df = self._load("data_mutations_extended.txt")
        counts = df["Hugo_Symbol"].value_counts()
        return counts

    def list_variant_classification(self):
        df = self._load("data_mutations_extended.txt")
        counts = df["Variant_Classification"].value_counts()
        return counts

    def list_sample_types(self):
        df = self._load("data_clinical_sample.txt")
        counts = df["SAMPLE_TYPE"].value_counts()
        return counts

    def list_genes_in_panel(self, seq_assay_id=None):
        genomic = self._load("genomic_information.txt")

        available_panels = genomic["SEQ_ASSAY_ID"].values
        if seq_assay_id not in available_panels:
            raise ValueError(f"Panel '{seq_assay_id}' not found. \n"
                             f" Use list_panels() to see available panels.")

        panel_genes = (genomic[genomic["SEQ_ASSAY_ID"] == seq_assay_id]
                       ["Hugo_Symbol"]
                       .dropna()
                       .unique()
                       .tolist())
        panel_genes = sorted(panel_genes)

        print(f"Panel: {seq_assay_id}")
        print(f"Genes covered: {len(panel_genes)}")

        return panel_genes

    def get_panel_size(self, seq_assay_id, min_size=0.5):
        genomic = self._load("genomic_information.txt")

        available_panels = set(genomic["SEQ_ASSAY_ID"])
        # print(available_panels)
        if seq_assay_id not in available_panels:
            raise ValueError(f"Panel '{seq_assay_id}' not found. \n"
                             f" Use list_panels() to see available panels.")

        panel_df = genomic[
            (genomic["SEQ_ASSAY_ID"] == seq_assay_id) &
            (genomic["includeInPanel"] == True)
            ].copy()

        panel_df["region_size"] = (panel_df["End_Position"] -
                                   panel_df["Start_Position"]) + 1
        size_mb = panel_df["region_size"].sum() / 1_000_000

        print(f"Panel: {seq_assay_id}")
        print(f"Size: {size_mb:.3f} Mb")

        if size_mb < min_size:
            print(f"    Warning: panel size {size_mb:.3f} Mb is below "
                  f"minimum threshold {min_size} Mb — TMB may not be reliable")
            return None

        return round(size_mb, 3)

    def query_by_gene(self, genes, variant_classification=None, variant_type=None):
        df = self._load("data_mutations_extended.txt")

        if isinstance(genes, str):
            genes = [genes]

        genes = [g.upper() for g in genes]

        gene_df = df[df["Hugo_Symbol"].str.upper().isin(genes)]
        tsb = gene_df["Tumor_Sample_Barcode"]

        if len(tsb) == 0:
            raise ValueError(f"No mutations found for gene: {genes}")

        df = df[df["Tumor_Sample_Barcode"].isin(tsb)]

        if variant_classification:
            valid = df["Variant_Classification"].unique().tolist()
            if variant_classification not in valid:
                raise ValueError(f"'{variant_classification}' not valid. Valid options: {valid}")
            df = df[df["Variant_Classification"] == variant_classification]

        if variant_type:
            valid = df["Variant_Type"].unique().tolist()
            if variant_type not in valid:
                raise ValueError(f"'{variant_type}' not valid. Valid options: {valid}")
            df = df[df["Variant_Type"] == variant_type]

        n_samples = df["Tumor_Sample_Barcode"].nunique()
        print(f"Found {n_samples} samples")

        return df.reset_index(drop=True)

    def query_by_sample(self, sample_id=None, patient_id=None):
        df = self._load("data_mutations_extended.txt")

        if sample_id is None and patient_id is None:
            raise ValueError("Please provide either a sample_id or patient_id")

        if sample_id:
            df = df[df["Tumor_Sample_Barcode"] == sample_id]

        if patient_id:
            samples = self._load("data_clinical_sample.txt")
            patient_samples = samples[samples["PATIENT_ID"] == patient_id]["SAMPLE_ID"]
            df = df[df["Tumor_Sample_Barcode"].isin(patient_samples)]

        if len(df) == 0:
            raise ValueError(f"No mutations found")

        n_samples = df["Tumor_Sample_Barcode"].nunique()
        print(f"Found {len(df)} mutations across {n_samples} samples")

        return df.reset_index(drop=True)

    def query_by_mutation(self, gene=None, mutation_id=None):
        df = self._load("data_mutations_extended.txt")

        if gene is None and mutation_id is None:
            raise ValueError("Please provide at least a gene or mutation_id")

        if gene and mutation_id:
            if not mutation_id.startswith("p."):
                mutation_id = "p." + mutation_id
            df = df[(df["Hugo_Symbol"].str.upper() == gene.upper()) &
                    (df["HGVSp_Short"] == mutation_id)]

        elif gene:
            df = df[df["Hugo_Symbol"].str.upper() == gene.upper()]

        elif mutation_id:
            if not mutation_id.startswith("p."):
                mutation_id = "p." + mutation_id
            df = df[df["HGVSp_Short"] == mutation_id]

        if len(df) == 0:
            raise ValueError(f"No mutations found for gene={gene}, mutation_id={mutation_id}")

        print(f"Found {len(df)} mutations across {df['Tumor_Sample_Barcode'].nunique()} samples")

        return df.reset_index(drop=True)

    def query_mutations_by_cohort(self, cancer_type=None, oncotree_code=None,
                                  center=None, sex=None, age_range=None,
                                  sample_type=None):

        mutations = self._load("data_mutations_extended.txt")

        # get cohort samples
        samples = self.query_samples(
            cancer_type=cancer_type,
            oncotree_code=oncotree_code,
            sample_type=sample_type
        )

        if center or sex or age_range:
            patients = self.query_patients(
                center=center,
                sex=sex)
            samples = samples[samples["PATIENT_ID"].isin(patients["PATIENT_ID"])]

        valid_samples = samples["SAMPLE_ID"].unique()
        df = mutations[mutations["Tumor_Sample_Barcode"].isin(valid_samples)]

        print(f"Found {len(df)} mutations across {df['Tumor_Sample_Barcode'].nunique()} samples")

        return df.reset_index(drop=True)

    def query_multiple_tumors(self, patient_id=None):
        samples = self._load("data_clinical_sample.txt")
        patients = self._load("data_clinical_patient.txt")

        # find patients with more than one sample
        samples_per_patient = samples.groupby("PATIENT_ID").size()
        multi_tumor_ids = samples_per_patient[samples_per_patient > 1].index

        # get their samples
        df = samples[samples["PATIENT_ID"].isin(multi_tumor_ids)]

        # join with patient info
        df = df.merge(patients, on="PATIENT_ID", how="left")

        # if a specific patient is requested
        if patient_id:
            df = df[df["PATIENT_ID"] == patient_id]
            if len(df) == 0:
                raise ValueError(f"No multiple tumor samples found for patient {patient_id}")

        # add a count column showing how many samples each patient has
        df["n_samples"] = df["PATIENT_ID"].map(samples_per_patient)

        print(f"Found {df['PATIENT_ID'].nunique()} patients with multiple tumors")
        print(f"Total samples: {len(df)}")

        return df.sort_values(["PATIENT_ID", "SAMPLE_ID"]).reset_index(drop=True)

    def query_fusions(self, gene=None, frame=None, dna_support=None, rna_support=None):
        df = self._load("data_fusions.txt")

        # Fix inconsistent formatting
        df["Frame"] = df["Frame"].str.lower().str.strip()
        df["Frame"] = df["Frame"].replace("in-frame", "in frame")
        df["Frame"] = df["Frame"].replace("unkown", "unknown")

        if gene:
            if isinstance(gene, str):
                gene = [gene]
            gene = [g.upper() for g in gene]
            df = df[df["Hugo_Symbol"].str.upper().isin(gene)]

        if frame:
            valid = ["in frame", "out of frame", "frameshift", "unknown"]
            if frame not in valid:
                raise ValueError(f"'{frame}' not valid. Valid options: {valid}")
            df = df[df["Frame"] == frame]

        if dna_support:
            df = df[df["DNA_support"] == dna_support]

        if rna_support:
            df = df[df["RNA_support"] == rna_support]

        if len(df) == 0:
            raise ValueError("No fusions found for the given filters")
        print(f"Found {len(df)} fusions across {df['Tumor_Sample_Barcode'].nunique()} samples")

        return df.reset_index(drop=True)

    def query_cna(self, gene=None, sample_id=None, cna_value=None):
        df = self._load("data_CNA.txt")

        # valid CNA values including -1.5
        valid_values = [-2, -1.5, -1, 0, 1, 2]
        cna_labels = {
            -2: "Deep deletion",
            -1.5: "Shallow/Deep deletion borderline",
            -1: "Shallow deletion",
            0: "Diploid",
            1: "Gain",
            2: "Amplification"
        }

        if cna_value is not None and cna_value not in valid_values:
            raise ValueError(f"'{cna_value}' not valid. Valid options: {valid_values}")

        if gene:
            if isinstance(gene, str):
                gene = [gene]
            gene = [g.upper() for g in gene]
            df = df[df["Hugo_Symbol"].str.upper().isin(gene)]

            if len(df) == 0:
                raise ValueError(f"No CNA data found for gene: {gene}")

        # melt from wide to long format
        df = df.melt(id_vars="Hugo_Symbol", var_name="SAMPLE_ID", value_name="cna_value")
        df = df.dropna(subset=["cna_value"])

        if sample_id:
            df = df[df["SAMPLE_ID"] == sample_id]

        if cna_value is not None:
            df = df[df["cna_value"] == cna_value]

        # add human readable label
        df["cna_label"] = df["cna_value"].map(cna_labels)

        if len(df) == 0:
            raise ValueError("No CNA data found for the given filters")

        print(f"Found {len(df)} CNA entries across {df['SAMPLE_ID'].nunique()} samples")

        return df.reset_index(drop=True)

    def build_cohort(self, cancer_type=None, oncotree_code=None, center=None,
                     sex=None, age_range=None, sample_type=None, include_mutations=False):

        patients = self.query_patients(center=center, sex=sex)
        samples = self.query_samples(cancer_type=cancer_type, oncotree_code=oncotree_code,
                                     age_range=age_range, sample_type=sample_type)

        cohort = samples.merge(patients, on="PATIENT_ID", how="inner")

        if include_mutations:
            mutations = self._load("data_mutations_extended.txt")
            cohort = cohort.merge(mutations, left_on="SAMPLE_ID",
                                  right_on="Tumor_Sample_Barcode", how="left")

        print(f"Cohort built: {len(cohort)} samples from {cohort['PATIENT_ID'].nunique()} patients")

        return cohort.reset_index(drop=True)

    def compute_normalized_tmb(self):
        mutations = self._load("data_mutations_extended.txt")
        samples = self._load("data_clinical_sample.txt")

        # Step 1: count mutations per sample
        mut_counts = mutations.groupby("Tumor_Sample_Barcode").size().reset_index(name="mutation_count")

        # Step 2: attach the sequencing assay to each sample
        mut_counts = mut_counts.merge(samples[["SAMPLE_ID", "SEQ_ASSAY_ID"]],
                                      left_on="Tumor_Sample_Barcode",
                                      right_on="SAMPLE_ID", how="left")

        # Step 3: calculate median mutation count per assay
        assay_medians = mut_counts.groupby("SEQ_ASSAY_ID")["mutation_count"].median()
        assay_medians.name = "assay_median"

        # Step 4: normalize each sample by its assay median
        mut_counts = mut_counts.merge(assay_medians, on="SEQ_ASSAY_ID", how="left")
        mut_counts["normalized_tmb"] = mut_counts["mutation_count"] / mut_counts["assay_median"]

        print(
            f"Computed normalized TMB for {len(mut_counts)} samples across {mut_counts['SEQ_ASSAY_ID'].nunique()} assays")

        return mut_counts.reset_index(drop=True)

    def compute_tmb(self, df=None, coding_only=True, min_panel_size=1):
        if df is not None:
            mutations = df
        else:
            mutations = self._load("data_mutations_extended.txt")

        samples = self._load("data_clinical_sample.txt")
        genomic = self._load("genomic_information.txt")

        # calculate panel sizes
        genomic_filtered = genomic[genomic["includeInPanel"] == True].copy()
        genomic_filtered["region_size"] = (genomic_filtered["End_Position"] - genomic_filtered["Start_Position"]) + 1
        panel_sizes = genomic_filtered.groupby("SEQ_ASSAY_ID")["region_size"].sum() / 1_000_000

        # filter out panels that are too small for reliable TMB
        small_panels = panel_sizes[panel_sizes < min_panel_size].index.tolist()
        if small_panels:
            print(f"Excluding {len(small_panels)} panels smaller than {min_panel_size} Mb: {small_panels}")
        panel_sizes = panel_sizes[panel_sizes >= min_panel_size]

        # keep only coding mutations if requested
        if coding_only:
            coding = ["Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del",
                      "Frame_Shift_Ins", "In_Frame_Del", "In_Frame_Ins",
                      "Splice_Site", "Translation_Start_Site", "Nonstop_Mutation"]
            mutations = mutations[mutations["Variant_Classification"].isin(coding)]

        # count mutations per sample
        mut_counts = mutations.groupby("Tumor_Sample_Barcode").size().reset_index(name="mutation_count")

        # attach panel info
        mut_counts = mut_counts.merge(
            samples[["SAMPLE_ID", "SEQ_ASSAY_ID"]],
            left_on="Tumor_Sample_Barcode",
            right_on="SAMPLE_ID",
            how="left"
        )

        # attach panel size
        mut_counts = mut_counts.merge(
            panel_sizes.reset_index(),
            on="SEQ_ASSAY_ID",
            how="left"
        )

        # compute TMB where possible
        mut_counts["tmb"] = mut_counts.apply(
            lambda r: r["mutation_count"] / r["region_size"]
            if pd.notna(r["region_size"]) and r["region_size"] >= min_panel_size else None, axis=1
        )

        # flag TMB high and hypermutators
        mut_counts["tmb_high"] = mut_counts["tmb"].apply(
            lambda x: True if x is not None and x >= 10 else False
        )
        mut_counts["tmb_hypermutator"] = mut_counts["tmb"].apply(
            lambda x: True if x is not None and x >= 100 else False
        )

        mut_counts["tmb_available"] = mut_counts["tmb"].notna()

        n_computed = mut_counts["tmb_available"].sum()
        n_total = len(mut_counts)
        n_high = mut_counts["tmb_high"].sum()
        n_hyper = mut_counts["tmb_hypermutator"].sum()

        print(f"TMB computed for {n_computed}/{n_total} samples")
        print(f"Could not compute TMB for {n_total - n_computed} samples (panel size unavailable or too small)")
        print(f"TMB-high (>=10 mut/Mb): {n_high} samples ({round(100 * n_high / n_computed, 1)}% of computed)")
        print(f"Hypermutators (>=100 mut/Mb): {n_hyper} samples ({round(100 * n_hyper / n_computed, 1)}% of computed)")

        return mut_counts.reset_index(drop=True)

    def run_sigprofiler(self, output_dir, cancer_type=None, oncotree_code=None,
                        center=None, cosmic_version=3.4, genome_build="GRCh37",
                        min_mutations=50):
        # python -c "from SigProfilerMatrixGenerator import install as genInstall; genInstall.install('GRCh37', rsync=False, bash=True)"
        import os
        import shutil
        from SigProfilerMatrixGenerator.scripts import SigProfilerMatrixGeneratorFunc as matGen
        from SigProfilerAssignment import Analyzer as Analyze

        # Step 1: get mutations for the cohort
        mutations = self._load("data_mutations_extended.txt")

        if cancer_type or oncotree_code or center:
            samples = self.query_samples(
                cancer_type=cancer_type,
                oncotree_code=oncotree_code
            )
            if center:
                patients = self.query_patients(center=center)
                samples = samples[samples["PATIENT_ID"].isin(patients["PATIENT_ID"])]

            valid_samples = samples["SAMPLE_ID"].unique()
            mutations = mutations[mutations["Tumor_Sample_Barcode"].isin(valid_samples)]

        # Step 2: filter samples with too few mutations
        mut_counts = mutations.groupby("Tumor_Sample_Barcode").size()
        valid_samples = mut_counts[mut_counts >= min_mutations].index
        excluded = mut_counts[mut_counts < min_mutations]
        mutations = mutations[mutations["Tumor_Sample_Barcode"].isin(valid_samples)]

        print(f"Samples with >= {min_mutations} mutations: {len(valid_samples)}")
        print(f"Samples excluded (too few mutations): {len(excluded)}")

        if len(valid_samples) == 0:
            raise ValueError(f"No samples with >= {min_mutations} mutations. Try lowering min_mutations.")

        # Step 3: prepare MAF in SigProfiler format
        maf = mutations.copy()
        maf["NCBI_Build"] = genome_build

        maf = maf.rename(columns={
            "Hugo_Symbol": "Hugo",
            "Entrez_Gene_Id": "Entrez",
            "Center": "Center",
            "NCBI_Build": "Genome",
            "Chromosome": "Chrom",
            "Start_Position": "Start",
            "End_Position": "End",
            "Strand": "Strand",
            "Variant_Classification": "Classification",
            "Variant_Type": "Type",
            "Reference_Allele": "Ref",
            "Tumor_Seq_Allele1": "Alt1",
            "Tumor_Seq_Allele2": "Alt2",
            "dbSNP_RS": "dbSNP",
            "dbSNP_Val_Status": "SNP_Val_status",
            "Tumor_Sample_Barcode": "Tumor_sample",
        })

        sigprofiler_cols = ["Hugo", "Entrez", "Center", "Genome", "Chrom",
                            "Start", "End", "Strand", "Classification", "Type",
                            "Ref", "Alt1", "Alt2", "dbSNP", "SNP_Val_status",
                            "Tumor_sample"]
        maf = maf[sigprofiler_cols]

        # Step 4: clean output_dir and save MAF
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        maf_path = os.path.join(output_dir, "genie.maf")
        maf.to_csv(maf_path, sep="\t", index=False)
        print(f"MAF saved: {len(maf)} mutations, {maf['Tumor_sample'].nunique()} samples")

        # Step 5: generate mutation matrix
        print("Generating mutation matrix...")
        matrices = matGen.SigProfilerMatrixGeneratorFunc(
            "genie",
            genome_build,
            output_dir,
            exome=True,
            bed_file=None,
            chrom_based=False,
            plot=False,
            seqInfo=True
        )

        # Step 6: run cosmic fit
        print("Running SigProfiler cosmic fit...")
        sbs_path = os.path.join(output_dir, "output", "SBS", "genie.SBS96.exome")

        Analyze.cosmic_fit(
            samples=sbs_path,
            output=os.path.join(output_dir, "sigprofiler_output"),
            input_type="matrix",
            context_type="96",
            cosmic_version=cosmic_version,
            genome_build=genome_build,
            collapse_to_SBS96=True,
            make_plots=False,
            sample_reconstruction_plots=False,
            verbose=True
        )

        print(f"Done! Results saved to {output_dir}/sigprofiler_output")

    def export_for_musical(self, sigprofiler_output_dir):
        import os

        matrix_path = os.path.join(sigprofiler_output_dir, "output", "SBS", "genie.SBS96.exome")

        if not os.path.exists(matrix_path):
            raise FileNotFoundError(
                f"SigProfiler matrix not found at {matrix_path}\n"
                f"Please run run_sigprofiler() first."
            )

        print(f"Matrix ready for MuSiCal at:\n{matrix_path}")
        print(f"\nFor MuSiCal usage see the full pipeline notebook:")
        print(f"https://github.com/parklab/MuSiCal/blob/main/examples/example_full_pipeline.ipynb")

        return matrix_path

    def export_for_signature_analyzer(self, output_dir, cancer_type=None, oncotree_code=None,
                                      center=None, sex=None, age_range=None,
                                      sample_type=None, min_mutations=50):
        import os

        mutations = self._load("data_mutations_extended.txt")

        if cancer_type or oncotree_code or center or sex or age_range or sample_type:
            samples = self.query_samples(
                cancer_type=cancer_type,
                oncotree_code=oncotree_code,
                sample_type=sample_type
            )
            if center or sex or age_range:
                patients = self.query_patients(
                    center=center,
                    sex=sex,
                    age_range=age_range
                )
                samples = samples[samples["PATIENT_ID"].isin(patients["PATIENT_ID"])]

            valid_samples = samples["SAMPLE_ID"].unique()
            mutations = mutations[mutations["Tumor_Sample_Barcode"].isin(valid_samples)]

        mut_counts = mutations.groupby("Tumor_Sample_Barcode").size()
        valid_samples = mut_counts[mut_counts >= min_mutations].index
        excluded = mut_counts[mut_counts < min_mutations]
        mutations = mutations[mutations["Tumor_Sample_Barcode"].isin(valid_samples)]

        print(f"Samples with >= {min_mutations} mutations: {len(valid_samples)}")
        print(f"Samples excluded (too few mutations): {len(excluded)}")

        if len(valid_samples) == 0:
            raise ValueError(f"No samples with >= {min_mutations} mutations. Try lowering min_mutations.")

        sa_cols = ["Hugo_Symbol", "Tumor_Sample_Barcode", "Chromosome",
                   "Start_Position", "Reference_Allele", "Tumor_Seq_Allele2",
                   "Variant_Type"]


        sa_muts = mutations[sa_cols].reset_index(drop=True)

        sa_muts["Chromosome"] = sa_muts["Chromosome"].astype(str)
        sa_muts["Chromosome"] = sa_muts["Chromosome"].apply(
            lambda x: x if x.startswith("chr") else "chr" + x
        )

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "signature_analyzer.maf")
        sa_muts.to_csv(output_path, sep="\t", index=False)

        print(f"MAF saved: {len(sa_muts)} mutations, {sa_muts['Tumor_Sample_Barcode'].nunique()} samples")
        print(f"File saved to: {output_path}")

        return sa_muts

    def plot_cancer_type_distribution(self, df=None, color="navy",top_n=20, save_path=None):
        import matplotlib.pyplot as plt

        # if not df passed use full dataset
        if df is None:
            df = self._load("data_clinical_sample.txt")
            count_col = "CANCER_TYPE"

        else:
            # df is from query
            samples = self._load("data_clinical_sample.txt")
            df = df.merge(samples[["SAMPLE_ID", "CANCER_TYPE"]],
                          left_on="Tumor_Sample_Barcode",
                          right_on="SAMPLE_ID", how="left")
            count_col = "CANCER_TYPE"

        counts = df[count_col].value_counts()
        counts = counts[counts.index != "UNKNOWN"].head(top_n)

        fig, ax = plt.subplots(figsize=(8,6))
        ax.bar(counts.index, counts.values, color = color)
        ax.set_xlabel("Cancer Type", fontsize=14)
        ax.set_ylabel("Number of Samples", fontsize=14)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved to {save_path}")

        plt.show()
        return counts

    def plot_age_distribution(self, df=None, color="navy", bins=20, save_path=None):
        import matplotlib.pyplot as plt
        if df is None:
            df = self._load("data_clinical_sample.txt")
        else:
            samples = self._load("data_clinical_sample.txt")
            df = df.merge(samples[["SAMPLE_ID", "AGE_AT_SEQ_REPORT"]],
                          left_on="Tumor_Sample_Barcode",
                          right_on="SAMPLE_ID", how="left")

        df["AGE_AT_SEQ_REPORT"] = pd.to_numeric(df["AGE_AT_SEQ_REPORT"], errors="coerce")
        ages = df["AGE_AT_SEQ_REPORT"].dropna()

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(ages, bins=bins, color=color, edgecolor="white")
        ax.set_xlabel("Age at Sequencing", fontsize=14)
        ax.set_ylabel("Number of Samples", fontsize=14)
        ax.axvline(ages.median(), color="black", linestyle="--",
                   label=f"Median: {ages.median():.1f}")
        ax.legend()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved to {save_path}")
        plt.show()

        print(f"Mean age: {ages.mean():.1f}")
        print(f"Median age: {ages.median():.1f}")
        print(f"Range: {ages.min():.0f} - {ages.max():.0f}")

        return ages


    def plot_center_distribution(self, df=None, color="navy", save_path=None):
        import matplotlib.pyplot as plt
        if df is None:
            df = self._load("data_clinical_patient.txt")
        else:
            # join with patient to get center
            patients = self._load("data_clinical_patient.txt")
            samples = self._load("data_clinical_sample.txt")
            df = df.merge(samples[["SAMPLE_ID", "PATIENT_ID"]],
                          left_on="Tumor_Sample_Barcode",
                          right_on="SAMPLE_ID", how="left")
            df = df.merge(patients[["PATIENT_ID", "CENTER"]],
                          on="PATIENT_ID", how="left")

        counts = df["CENTER"].value_counts()

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(counts.index, counts.values, color=color)
        ax.set_xlabel("Center", fontsize=14)
        ax.set_ylabel("Number of Samples", fontsize=14)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Plot saved to {save_path}")

        plt.show()
        return counts

    def plot_mutation_burden(self, df=None, method="tmb", color="navy",
                             bins=50, log_scale=True, save_path=None):
        import matplotlib.pyplot as plt

        # method can be "tmb" or "normalized"
        if method not in["tmb", "normalized"]:
            raise ValueError("Method must be 'tmb' or 'normalized'")

        if method == "tmb":
            tmb_df = self.compute_tmb(df)
            tmb_df = tmb_df[tmb_df["tmb_available"] == True]
            values = tmb_df["tmb"]
            xlabel = "TMB (mutation/Mb)"
        else:
            tmb_df = self.compute_normalized_tmb()
            values = tmb_df["normalized_tmb"]
            xlabel = "Normalized TMB (relative to panel median)"

        # filter to specific samples if df provided
        if df is not None:
            valid_samples = df["Tumor_Sample_Barcode"].unique()
            tmb_df = tmb_df[tmb_df["Tumor_Sample_Barcode"].isin(valid_samples)]
            values = tmb_df["tmb"] if method == "tmb" else tmb_df["normalized_tmb"]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(values, bins=bins, color=color, edgecolor="white")

        if log_scale:
            ax.set_yscale("log")
            ax.set_ylabel("$\log_{10}$(Number of Samples)", fontsize=14)
        else:
            ax.set_ylabel("Number of Samples", fontsize=14)


        ax.set_xlabel(xlabel, fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Plot saved to {save_path}")

        plt.show()

        print(f"Samples included: {len(values)}")
        print(f"Median: {values.median():.2f}")
        print(f"Mean: {values.mean():.2f}")
        print(f"Max: {values.max():.2f}")

        return values

    def plot_variant_classification(self, df=None, color="navy", save_path=None):
        import matplotlib.pyplot as plt

        if df is None:
            df = self._load("data_mutations_extended.txt")
        else:
            # if df comes from query_by_gene or query_by_mutation
            # it already has Variant_Classification column so no join needed
            pass

        counts = df["Variant_Classification"].value_counts()

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(counts.index, counts.values, color=color)
        ax.set_xlabel("Variant Classification", fontsize=14)
        ax.set_ylabel("Count", fontsize=14)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Plot saved to {save_path}")

        plt.show()
        return counts

    def plot_top_mutated_genes(self, df=None, top_n=20, color="navy", save_path=None):
        import matplotlib.pyplot as plt
        if df is None:
            df = self._load("data_mutations_extended.txt")

        n_samples = df["Tumor_Sample_Barcode"].nunique()
        if n_samples == 1:
            raise ValueError("Only 1 sample is found. Maybe this is not the function for you.")

        gene_counts = df.groupby("Hugo_Symbol")["Tumor_Sample_Barcode"].nunique().sort_values(ascending=False).head(top_n)

        gene_freq = (gene_counts / n_samples * 100).round(1)

        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.barh(gene_counts.index[::-1], gene_freq.values[::-1], color=color)

        ax.set_xlabel("% Samples Mutated", fontsize=12)
        ax.set_ylabel("Gene", fontsize=12)

        # add percentage labels on bars
        for bar, freq in zip(bars, gene_freq.values[::-1]):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{freq}%", va="center", fontsize=9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Plot saved to {save_path}")

        plt.show()

        print(f"Based on {n_samples} samples")
        return gene_counts

    def plot_signature_exposure(self, sigprofiler_output_dir, top_n_signatures=10,
                                top_n_samples=50, all_samples=False, save_path=None):
        import matplotlib.pyplot as plt
        import numpy as np
        import os

        activities_path = os.path.join(
            sigprofiler_output_dir, "sigprofiler_output",
            "Assignment_Solution", "Activities",
            "Assignment_Solution_Activities.txt"
        )

        if not os.path.exists(activities_path):
            raise FileNotFoundError(
                f"Activities file not found at {activities_path}\n"
                f"Please run run_sigprofiler() first."
            )

        df = pd.read_csv(activities_path, sep="\t", index_col=0)

        # keep only signatures with non-zero activity
        df = df.loc[:, (df > 0).any(axis=0)]

        # get top N signatures by total activity
        top_sigs = df.sum(axis=0).sort_values(ascending=False).head(top_n_signatures).index.tolist()
        df = df[top_sigs]

        # all samples or top N
        if all_samples:
            df = df.loc[df.sum(axis=1).sort_values(ascending=False).index]
            print(f"Showing all {len(df)} samples")
        else:
            df = df.loc[df.sum(axis=1).sort_values(ascending=False).head(top_n_samples).index]
            print(f"Showing top {top_n_samples} of {len(df)} samples")

        # normalize each sample to 100%
        df_norm = df.div(df.sum(axis=1), axis=0) * 100

        # plot stacked bar
        fig, ax = plt.subplots(figsize=(16, 7))

        colors = plt.cm.tab20.colors
        bottom = np.zeros(len(df_norm))

        for i, sig in enumerate(top_sigs):
            ax.bar(range(len(df_norm)), df_norm[sig],
                   bottom=bottom, label=sig,
                   color=colors[i % len(colors)], width=0.8)
            bottom += df_norm[sig].values

        ax.set_xlabel("Samples", fontsize=14)
        ax.set_ylabel("Signature Contribution (%)", fontsize=14)
        ax.set_xticks([])
        ax.legend(loc="upper right", bbox_to_anchor=(1.12, 1), fontsize=9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Plot saved to {save_path}")

        plt.show()
        return df_norm

    def plot_mutational_profile(self, sigprofiler_output_dir, sample_id, save_path=None):
        import matplotlib.pyplot as plt
        import os

        sbs_path = os.path.join(sigprofiler_output_dir, "output", "SBS", "genie.SBS96.exome")

        if not os.path.exists(sbs_path):
            raise FileNotFoundError(
                f"SBS96 matrix not found at {sbs_path}\n"
                f"Please run run_sigprofiler() first."
            )

        df = pd.read_csv(sbs_path, sep="\t", index_col=0)

        if sample_id not in df.columns:
            raise ValueError(
                f"Sample '{sample_id}' not found.\n"
                f"Available samples: {df.columns.tolist()[:5]}..."
            )

        counts = df[sample_id]

        # define 6 mutation types and their colors
        mutation_types = {
            "C>A": "#1EBFF0",
            "C>G": "#050708",
            "C>T": "#E62725",
            "T>A": "#CBCACB",
            "T>C": "#A1CE63",
            "T>G": "#EDC8C5",
        }

        fig, ax = plt.subplots(figsize=(20, 4))

        values = []
        bar_colors = []
        xtick_labels = []

        for mut_type, color in mutation_types.items():
            channels = [c for c in counts.index if f"[{mut_type}]" in c]
            for channel in channels:
                values.append(counts[channel])
                bar_colors.append(color)
                five_prime = channel[0]
                ref = channel[2]
                three_prime = channel[-1]
                xtick_labels.append(f"{five_prime}{ref}{three_prime}")

        ax.bar(range(len(values)), values, color=bar_colors, width=0.8)

        # add mutation type labels and shading
        x_pos = 0
        for mut_type, color in mutation_types.items():
            channels = [c for c in counts.index if f"[{mut_type}]" in c]
            n = len(channels)
            ax.axvspan(x_pos - 0.5, x_pos + n - 0.5, alpha=0.1, color=color)
            ax.text(x_pos + n / 2 - 0.5, max(values) * 1.08, mut_type,
                    ha="center", fontsize=11, fontweight="bold", color=color)
            if x_pos > 0:
                ax.axvline(x_pos - 0.5, color="white", linewidth=2)
            x_pos += n

        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(xtick_labels, rotation=90, fontsize=14, fontfamily="monospace")
        ax.set_ylabel("Mutation Count", fontsize=14)
        ax.set_xlim(-0.5, len(values) - 0.5)
        plt.subplots_adjust(bottom=0.2)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Plot saved to {save_path}")

        plt.show()
        return counts

    def comutation(self, genes, oncotree_code=None, cancer_type=None,
                   center=None, save_plot=False, save_path=None):
        from scipy.stats import fisher_exact
        import matplotlib.pyplot as plt
        import numpy as np
        from itertools import combinations

        if len(genes) < 2:
            raise ValueError("Please provide at least 2 genes")

        # get cohort mutations
        mutations = self.query_mutations_by_cohort(
            oncotree_code=oncotree_code,
            cancer_type=cancer_type,
            center=center
        )

        # get total samples in cohort
        samples = self.query_samples(
            oncotree_code=oncotree_code,
            cancer_type=cancer_type
        )
        if center:
            patients = self.query_patients(center=center)
            samples = samples[samples["PATIENT_ID"].isin(patients["PATIENT_ID"])]

        all_samples = set(samples["SAMPLE_ID"].unique())
        n_total = len(all_samples)

        # builg binary mutation matrix (sample x gene)
        genes_upper = [g.upper() for g in genes]
        mut_pivot = (
            mutations.assign(Hugo_upper=mutations["Hugo_Symbol"].str.upper())
            .groupby(["Tumor_Sample_Barcode", "Hugo_upper"])
            .size()
            .unstack(fill_value=0)
            .clip(upper=1)
        )

        # add missing genes as zeros
        for g in genes_upper:
            if g not in mut_pivot.columns:
                mut_pivot[g] = 0

        # samples with no mutations as zeros
        missing_samples = all_samples - set(mut_pivot.index)
        if missing_samples:
            zero_rows = pd.DataFrame(0, index=list(missing_samples), columns=mut_pivot.columns)
            mut_pivot = pd.concat([mut_pivot, zero_rows])

        # pairwise analysis
        def format_pvalue(p):
            if p < 0.0001:
                return "< 0.0001"
            elif p < 0.001:
                return "< 0.001"
            elif p < 0.01:
                return "< 0.01"
            elif p < 0.05:
                return "< 0.05"
            else:
                return str(round(p, 4))

        rows = []

        for gene_a, gene_b in combinations(genes_upper, 2):
            col_a = mut_pivot.get(gene_a, pd.Series(0, index=mut_pivot.index))
            col_b = mut_pivot.get(gene_b, pd.Series(0, index=mut_pivot.index))

            n_both = int(((col_a == 1) & (col_b == 1)).sum())
            n_a_only = int(((col_a == 1) & (col_b == 0)).sum())
            n_b_only = int(((col_a == 0) & (col_b == 1)).sum())
            n_neither = int(((col_a == 0) & (col_b == 0)).sum())

            freq_a = round((n_both + n_a_only) / n_total, 4)
            freq_b = round((n_both + n_b_only) / n_total, 4)
            freq_both = round(n_both / n_total, 4)

            contingency = [[n_both, n_a_only], [n_b_only, n_neither]]
            odds_ratio, p_value = fisher_exact(contingency)

            if p_value < 0.05:
                relationship = "co-mutation" if odds_ratio > 1 else "mutual exclusivity"
            else:
                relationship = "none"

            rows.append({
                "gene_a": gene_a, "gene_b": gene_b,
                "n_total": n_total,
                "n_both": n_both, "n_a_only": n_a_only,
                "n_b_only": n_b_only, "n_neither": n_neither,
                "freq_a": freq_a, "freq_b": freq_b, "freq_both": freq_both,
                "odds_ratio": round(odds_ratio, 4),
                "p_value": round(p_value, 6),
                "p_value_label": format_pvalue(p_value),
                "relationship": relationship
            })

        result = pd.DataFrame(rows).sort_values("p_value")

        # optional plot — heatmap of odds ratios
        if save_plot:
            matrix = pd.DataFrame(index=genes_upper, columns=genes_upper, dtype=float)
            for _, row in result.iterrows():
                matrix.loc[row["gene_a"], row["gene_b"]] = row["odds_ratio"]
                matrix.loc[row["gene_b"], row["gene_a"]] = row["odds_ratio"]
            for g in genes_upper:
                matrix.loc[g, g] = 1.0

            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(matrix.values.astype(float), cmap="RdBu_r", vmin=0, vmax=3)
            ax.set_xticks(range(len(genes_upper)))
            ax.set_yticks(range(len(genes_upper)))
            ax.set_xticklabels(genes_upper, ha="center", fontsize=14)
            ax.set_yticklabels(genes_upper, fontsize=14)
            plt.colorbar(im, ax=ax, label="Odds Ratio")
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                print(f"Plot saved to {save_path}")
            plt.show()

        print(f"\nCo-mutation results ({n_total} samples):")
        print(f"Co-mutations found: {(result['relationship'] == 'co-mutation').sum()}")
        print(f"Mutual exclusivities found: {(result['relationship'] == 'mutual exclusivity').sum()}")

        return result.reset_index(drop=True)

    def search_gene(self, query, top_n=10):
        import difflib
        df = self._load("data_mutations_extended.txt")
        all_genes = df["Hugo_Symbol"].unique().tolist()

        # exact match
        query_upper = query.upper()
        exact = [g for g in all_genes if g.upper() == query_upper]
        if exact:
            print(f"Exact match found: {exact[0]}")
            return exact

        # fuzzy match
        matches = difflib.get_close_matches(query.upper(),
                                            [g.upper() for g in all_genes],
                                            n=top_n, cutoff=0.6)

        upper_to_original = {g.upper(): g for g in all_genes}
        matches = [upper_to_original[m] for m in matches]

        if len(matches) == 0:
            print(f"No genes found matching '{query}'")
            return []
        print(f"Did you mean one of these?")
        for m in matches:
            print(f"\t{m}")

        return matches

    def patient_report(self, patient_id):
        print(f"Patient Report: {patient_id}")

        # clinical info
        patients = self._load("data_clinical_patient.txt")  # bug 1: was data_clinical_patients (extra s)
        patient = patients[patients["PATIENT_ID"] == patient_id]  # bug 2: was Patient_ID (wrong case)

        if len(patient) == 0:  # bug 3: was checking len(patients) not len(patient)
            raise ValueError(f"Patient '{patient_id}' not found")

        patient = patient.iloc[0]
        print("-------------Clinical Information-------------")
        print(f"    Sex: {patient['SEX']}")
        print(f"    Race: {patient['PRIMARY_RACE']}")
        print(f"    Center: {patient['CENTER']}")
        print(f"    Dead: {patient['DEAD']}")
        print(f"    Year contact: {patient['YEAR_CONTACT']}")
        print(f"    Year death: {patient['YEAR_DEATH']}")

        # sample info
        samples = self._load("data_clinical_sample.txt")
        patient_samples = samples[samples["PATIENT_ID"] == patient_id]  # bug 4: was Patient_ID

        print("-------------Samples Information-------------")
        print(f"    Samples: {len(patient_samples)}")

        # bug 5: was using 's' variable without a loop
        for _, s in patient_samples.iterrows():
            print(f"    Sample ID:    {s['SAMPLE_ID']}")
            print(f"    Cancer type:  {s['CANCER_TYPE']}")
            print(f"    OncotreeCode: {s['ONCOTREE_CODE']}")
            print(f"    Sample type:  {s['SAMPLE_TYPE']}")
            print(f"    Panel:        {s['SEQ_ASSAY_ID']}")
            print(f"    Age at seq:   {s['AGE_AT_SEQ_REPORT']}")
            print()

        # mutations
        mutations = self._load("data_mutations_extended.txt")
        patient_muts = mutations[mutations["Tumor_Sample_Barcode"].isin(
            patient_samples["SAMPLE_ID"]
        )]

        print("-------------Mutations Information-------------")
        print(f"    Total mutations: {len(patient_muts)}")  # bug 6: said "Samples" instead of mutations
        if len(patient_muts) > 0:
            for _, m in patient_muts.iterrows():
                print(f"    {m['Hugo_Symbol']} {m['HGVSp_Short']} "
                      f"({m['Variant_Classification']}) — {m['Tumor_Sample_Barcode']}")
        else:
            print("    No mutations found")

        # CNA
        try:
            cna_df = self._load("data_CNA.txt")
            patient_cols = [c for c in cna_df.columns
                            if c in patient_samples["SAMPLE_ID"].values]
            if patient_cols:
                cna_patient = cna_df[["Hugo_Symbol"] + patient_cols]
                cna_patient = cna_patient.melt(id_vars="Hugo_Symbol",
                                               var_name="SAMPLE_ID",
                                               value_name="cna_value")
                cna_patient = cna_patient[cna_patient["cna_value"] != 0].dropna()

                cna_labels = {-2: "Deep deletion", -1: "Shallow deletion",
                              -1.5: "Borderline deletion", 1: "Gain", 2: "Amplification"}

                print("-------------Copy Number Alterations-------------")
                print(f"    Total CNAs: {len(cna_patient)}")

                if len(cna_patient) > 0:
                    for _, c in cna_patient.iterrows():
                        label = cna_labels.get(c["cna_value"], str(c["cna_value"]))
                        print(f"    {c['Hugo_Symbol']} — {label} ({c['SAMPLE_ID']})")
                else:
                    print("    No CNAs found")
        except FileNotFoundError:
            pass

        # fusions
        try:
            fusions = self._load("data_fusions.txt")
            patient_fusions = fusions[fusions["Tumor_Sample_Barcode"].isin(
                patient_samples["SAMPLE_ID"]
            )]
            print("-------------Fusions-------------")
            print(f"    Total fusions: {len(patient_fusions)}")

            if len(patient_fusions) > 0:
                for _, f in patient_fusions.iterrows():
                    frame = f['Frame'] if pd.notna(f['Frame']) else "unknown frame"
                    print(f"    {f['Fusion']} — {frame} ({f['Tumor_Sample_Barcode']})")
            else:
                print("    No fusions found")
        except FileNotFoundError:
            pass


        return {
            "clinical": patient.to_dict(),
            "samples": patient_samples,
            "mutations": patient_muts,
        }

    def plot_lollipop_lollipops(self, gene, top_n=50, oncotree_code=None,
                                cancer_type=None, output_path=None,
                                mutations=None):
        """
        Generate a lollipop plot using the lollipops tool.

        Requires lollipops to be installed:
            go install github.com/joiningdata/lollipops@latest
            export PATH=$PATH:$(go env GOPATH)/bin

        Parameters
        ----------
        gene : str
            Gene symbol e.g. 'TP53'
        top_n : int
            Top N mutation positions to show (default 50)
        oncotree_code : str, optional
            Filter by OncotreeCode
        cancer_type : str, optional
            Filter by cancer type
        output_path : str, optional
            Output file path (.svg or .png). Default: {gene}_lollipop.svg
        mutations : list of str, optional
            Manually specify mutations. If provided, skips data loading.
            Accepted formats:
                ['R175H']                  — position only
                ['R175H@45']               — position + count
                ['R175H#ff6a6a']           — position + color
                ['R175H#ff6a6a@45']        — position + color + count
        """
        import subprocess
        import os

        if mutations is None:
            df = self.query_by_gene(gene)

            if oncotree_code or cancer_type:
                samples = self.query_samples(
                    oncotree_code=oncotree_code,
                    cancer_type=cancer_type
                )
                df = df[df["Tumor_Sample_Barcode"].isin(samples["SAMPLE_ID"])]

            coding = ["Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del",
                      "Frame_Shift_Ins", "In_Frame_Del", "In_Frame_Ins",
                      "Splice_Site", "Translation_Start_Site", "Nonstop_Mutation"]
            df = df[df["Variant_Classification"].isin(coding)]

            if len(df) == 0:
                raise ValueError(f"No coding mutations found for {gene}")

            color_map = {
                "Missense_Mutation": "ff6a6a",
                "Nonsense_Mutation": "fff68f",
                "Frame_Shift_Del": "00bfff",
                "Frame_Shift_Ins": "00bfff",
                "In_Frame_Del": "ff8c00",
                "In_Frame_Ins": "ff8c00",
                "Splice_Site": "ff7f50",
                "Translation_Start_Site": "ff7f50",
                "Nonstop_Mutation": "ff7f50",
            }

            mut_counts = (df.groupby(["HGVSp_Short", "Variant_Classification"])
                          .size()
                          .reset_index(name="count")
                          .nlargest(top_n, "count"))

            mutations = []
            for _, row in mut_counts.iterrows():
                hgvsp = str(row["HGVSp_Short"]).replace("p.", "")
                color = color_map.get(row["Variant_Classification"], "b0b0b0")
                mutations.append(f"{hgvsp}#{color}@{int(row['count'])}")

        if output_path is None:
            output_path = f"{gene}_lollipop.svg"

        lollipops_bin = os.path.expanduser("~/go/bin/lollipops")
        cmd = [lollipops_bin, "-legend", "-labels", f"-o={output_path}", gene] + mutations

        print(f"Generating lollipop plot for {gene} ({len(mutations)} positions)...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            print(f"Plot saved to {output_path}")
        else:
            raise RuntimeError(
                f"lollipops failed:\n{result.stderr}\n"
                f"Make sure lollipops is installed and in PATH."
            )

        return output_path

    def plot_age_survival(self, stratify_by=None, stratify_values=None,
                          oncotree_code=None, cancer_type=None,
                          center=None, gene=None, save_path=None):

        import matplotlib.pyplot as plt

        samples = self.query_samples(
            oncotree_code=oncotree_code,
            cancer_type=cancer_type
        )

        patients = self.query_patients(center=center)

        # join
        df = samples.merge(patients, on="PATIENT_ID", how="inner")

        # one row per patient
        df = df.drop_duplicates(subset="PATIENT_ID")

        if len(df) == 0:
            raise ValueError("No patients found for the given filters")

        # stratify by gene mutation status
        if gene:
            mutations = self._load("data_mutations_extended.txt")
            mutated_samples = set(
                mutations[mutations["Hugo_Symbol"].str.upper() == gene.upper()
                          ]["Tumor_Sample_Barcode"]
            )
            sample_to_patient = samples.set_index("SAMPLE_ID")["PATIENT_ID"].to_dict()
            mutated_patients = {sample_to_patient.get(s) for s in mutated_samples}
            df["_group"] = df["PATIENT_ID"].apply(
                lambda x: f"{gene} mutated" if x in mutated_patients else f"{gene} WT"
            )
        elif stratify_by:
            df["_group"] = df[stratify_by].astype(str)
            if stratify_values:
                df = df[df["_group"].isin([str(v) for v in stratify_values])]
        else:
            df["_group"] = "All patients"

        # prepare survival data
        df = df.copy()
        df["_time"] = pd.to_numeric(df["INT_CONTACT"], errors="coerce") / 365.25
        df["_event"] = df["DEAD"].apply(
            lambda x: 1 if str(x).upper() == "TRUE" else 0
        )
        df = df.dropna(subset=["_time"])
        df = df[df["_time"] > 0]

        if len(df) == 0:
            raise ValueError("No patients with valid survival data found")

        # kaplan meier estimator
        def kaplan_meier(times, events):
            data = sorted(zip(times, events), key=lambda x: x[0])
            S, results = 1.0, [(0, 1.0)]
            n = len(data)
            i = 0
            while i < n:
                t = data[i][0]
                d = sum(1 for tt, ee in data[i:] if tt == t and ee == 1)
                n_risk = n - i
                if d > 0:
                    S *= (1 - d / n_risk)
                results.append((t, S))
                while i < n and data[i][0] == t:
                    i += 1
            return zip(*results)

        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ["#2166AC", "#D6604D", "#4DAC26", "#984EA3", "#FF7F00"]
        groups = sorted(df["_group"].unique())

        for idx, group in enumerate(groups):
            grp = df[df["_group"] == group]
            times = grp["_time"].tolist()
            events = grp["_event"].tolist()
            km_times, km_surv = kaplan_meier(times, events)
            km_times, km_surv = list(km_times), list(km_surv)

            color = colors[idx % len(colors)]
            n = len(times)
            median_t = next((t for t, s in zip(km_times, km_surv) if s <= 0.5), None)
            label = f"{group} (n={n})"
            if median_t:
                label += f", median={median_t:.0f} years"

            ax.step(km_times, km_surv, where="post", color=color, linewidth=2, label=label)

        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Age at last contact (years)", fontsize=14)
        ax.set_ylabel("Survival Probability", fontsize=14)

        title = "Kaplan-Meier"
        if oncotree_code:
            title += f" — {oncotree_code}"
        if gene:
            title += f" — {gene}"
        ax.set_title(title, fontsize=13)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)

        # disclaimer
        ax.text(0.01, 0.02,
                "Note: x-axis is age at last contact, not time from diagnosis",
                transform=ax.transAxes, fontsize=7, color="gray", style="italic")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Plot saved to {save_path}")

        plt.show()

        return df[["PATIENT_ID", "_group", "_time", "_event"]].reset_index(drop=True)

    def run_sigma(self, output_dir, df=None, cancer_type=None, oncotree_code=None,
                  center=None,data="msk", catalog_name="cosmic_v2_inhouse",
                  check_msi=True, snv_cutoff=None):
        """
        SigMA is specifically designed for targeted panel data like GENIE,
        and is particularly sensitive for detecting Signature 3 (HR defect/BRCA).

        Requires R and SigMA installed:
            install.packages('devtools')
            devtools::install_github('parklab/SigMA')
        """

        import os
        import subprocess
        import shutil

        # Step 1: get mutations — use provided df or load from cohort
        if df is not None:
            mutations = df.copy()
            print(f"Using provided DataFrame: {len(mutations)} mutations")
        else:
            mutations = self._load("data_mutations_extended.txt")

            if cancer_type or oncotree_code or center:
                samples = self.query_samples(
                    cancer_type=cancer_type,
                    oncotree_code=oncotree_code
                )
                if center:
                    patients = self.query_patients(center=center)
                    samples = samples[samples["PATIENT_ID"].isin(patients["PATIENT_ID"])]
                valid_samples = samples["SAMPLE_ID"].unique()
                mutations = mutations[mutations["Tumor_Sample_Barcode"].isin(valid_samples)]


        os.makedirs(output_dir, exist_ok=True)
        maf_path = os.path.join(output_dir, "genie_sigma.maf")
        mutations.to_csv(maf_path, sep="\t", index=False)
        print(f"MAF saved: {len(mutations)} mutations, {mutations['Tumor_Sample_Barcode'].nunique()} samples")

        snv_cutoff_r = "NULL" if snv_cutoff is None else int(snv_cutoff)

        r_script = f"""
        library(SigMA)

        # load MAF and generate 96-channel matrix
        data_file <- '{maf_path}'
        genomes_matrix <- make_matrix(data_file, file_type='maf', ref_genome_name='hg19')
        genomes <- conv_snv_matrix_to_df(genomes_matrix)

        # save matrix
        genome_file <- '{os.path.join(output_dir, "sigma_matrix.csv")}'
        write.table(genomes, genome_file, sep=',', row.names=FALSE, 
                    col.names=TRUE, quote=FALSE)
        message(paste0('96-dimensional matrix saved to ', genome_file))

        # run SigMA
        message('Running SigMA...')
        output <- run(genome_file,
                      data='{data}',
                      do_assign=TRUE,
                      do_mva=TRUE,
                      lite_format=TRUE,
                      snv_cutoff={snv_cutoff_r},
                      check_msi={str(check_msi).upper()},
                      catalog_name='{catalog_name}')

        # save results
        output_file <- '{os.path.join(output_dir, "sigma_results.csv")}'
        message(paste0('SigMA results saved to ', output_file))
        """

        r_script_path = os.path.join(output_dir, "run_sigma.R")
        with open(r_script_path, "w") as f:
            f.write(r_script)
        print(f"R script saved to {r_script_path}")

        # Step 5: run R script
        rscript = shutil.which("Rscript")
        if rscript is None:
            raise FileNotFoundError(
                "Rscript not found. Install R from https://www.r-project.org/\n"
                f"R script saved to {r_script_path} — run manually with:\n"
                f"  Rscript {r_script_path}"
            )

        print("Running SigMA via Rscript...")
        result = subprocess.run(
            [rscript, r_script_path],
            capture_output=True, text=True, timeout=600
        )

        if result.returncode == 0:
            print(result.stdout)
            results_path = os.path.join(output_dir, "sigma_results.csv")
            print(f"Done! SigMA results saved to {results_path}")
            return pd.read_csv(results_path)
        else:
            print(result.stdout)
            raise RuntimeError(
                f"SigMA failed:\n{result.stderr}\n"
                f"Try running manually:\n  Rscript {r_script_path}"
            )


    def plot_sex_distribution(self, df=None, color="navy", save_path=None):
        import matplotlib.pyplot as plt

        if df is None:
            df = self._load("data_clinical_patient.txt")
        else:
            patients = self._load("data_clinical_patient.txt")
            samples = self._load("data_clinical_sample.txt")
            df = df.merge(samples[["SAMPLE_ID", "PATIENT_ID"]],
                          left_on="Tumor_Sample_Barcode",
                          right_on="SAMPLE_ID", how="left")
            df = df.merge(patients[["PATIENT_ID", "SEX"]],
                          on="PATIENT_ID", how="left")
        counts = df["SEX"].value_counts()
        counts = counts[~counts.index.isin(["Unknown", "Other"])]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(counts.index, counts.values, color=color)
        ax.set_xlabel("Sex", fontsize=14)
        ax.set_ylabel("Count", fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to {save_path}")

        plt.show()
        return counts

    def plot_race_distribution(self, df=None, color="navy", save_path=None):
        import matplotlib.pyplot as plt

        if df is None:
            df = self._load("data_clinical_patient.txt")
        else:
            patients = self._load("data_clinical_patient.txt")
            samples = self._load("data_clinical_sample.txt")
            df = df.merge(samples[["SAMPLE_ID", "PATIENT_ID"]],
                          left_on="Tumor_Sample_Barcode",
                          right_on="SAMPLE_ID", how="left")
            df = df.merge(patients[["PATIENT_ID", "PRIMARY_RACE"]],
                          on="PATIENT_ID", how="left")

        counts = df["PRIMARY_RACE"].value_counts()
        counts = counts[~counts.index.isin(["Unknown", "Not Collected",
                                            "Not Applicable", "Other"])]

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar(counts.index, counts.values, color=color)
        ax.set_xlabel("Race", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Plot saved to {save_path}")

        plt.show()
        return counts

    def query_by_center(self, center, include_mutations=False,
                        include_cna=False, include_fusions=False):
        patients = self.query_patients(center=center)
        samples = self.query_samples()
        samples = samples[samples["PATIENT_ID"].isin(patients["PATIENT_ID"])]

        print(f"Center: {center}")
        print(f"Patients: {len(patients)}")
        print(f"Samples: {len(samples)}")

        result = {
            "patients": patients,
            "samples": samples
        }

        if include_mutations:
            mutations = self._load("data_mutations_extended.txt")
            result["mutations"] = mutations[
                mutations["Tumor_Sample_Barcode"].isin(samples["SAMPLE_ID"])
            ].reset_index(drop=True)
            print(f"Mutations: {len(result['mutations'])}")

        if include_cna:
            cna = self._load("data_CNA.txt")
            center_cols = [c for c in cna.columns
                           if c in samples["SAMPLE_ID"].values]
            result["cna"] = cna[["Hugo_Symbol"] + center_cols]
            print(f"CNA samples: {len(center_cols)}")

        if include_fusions:
            fusions = self._load("data_fusions.txt")
            result["fusions"] = fusions[
                fusions["Tumor_Sample_Barcode"].isin(samples["SAMPLE_ID"])
            ].reset_index(drop=True)
            print(f"Fusions: {len(result['fusions'])}")

        return result

    def compare_cohorts(self, cohort1, cohort2, mutations_included=False,
                        cohort1_name="Cohort1", cohort2_name="Cohort2",
                        top_n_genes=20, save_path=None):
        from scipy.stats import fisher_exact
        import matplotlib.pyplot as plt

        def format_pvalue(p):
            if p < 0.0001:
                return "< 0.0001"
            elif p < 0.001:
                return "< 0.001"
            elif p < 0.01:
                return "< 0.01"
            elif p < 0.05:
                return "< 0.05"
            else:
                return str(round(p, 4))

        # get sample IDs for each cohort
        samples1 = cohort1["SAMPLE_ID"].unique()
        samples2 = cohort2["SAMPLE_ID"].unique()
        n1 = len(samples1)
        n2 = len(samples2)

        print(f"{cohort1_name}: {n1} samples")
        print(f"{cohort2_name}: {n2} samples")

        if not mutations_included:
            mutations = self._load("data_mutations_extended.txt")
            muts1 = mutations[mutations["Tumor_Sample_Barcode"].isin(samples1)]  # bug 1: typo Batcode
            muts2 = mutations[mutations["Tumor_Sample_Barcode"].isin(samples2)]  # bug 1: typo Batcode
        else:
            # if mutations included, cohort already has mutation columns
            # need to filter to only mutation rows (drop clinical-only rows)
            muts1 = cohort1[cohort1["Tumor_Sample_Barcode"].notna()]  # bug 2: was using cohort directly
            muts2 = cohort2[cohort2["Tumor_Sample_Barcode"].notna()]

        # bug 3: missing parenthesis in set union
        all_genes = set(muts1["Hugo_Symbol"].unique()) | set(muts2["Hugo_Symbol"].unique())

        # count mutated samples per gene
        counts1 = (muts1.groupby("Hugo_Symbol")["Tumor_Sample_Barcode"]
                   .nunique()
                   .reindex(all_genes, fill_value=0))
        counts2 = (muts2.groupby("Hugo_Symbol")["Tumor_Sample_Barcode"]
                   .nunique()
                   .reindex(all_genes, fill_value=0))

        # get top N by combined frequency
        top_genes = (counts1 / n1 + counts2 / n2).nlargest(top_n_genes).index.tolist()

        # statistical comparison
        rows = []
        for gene in top_genes:
            n_mut1 = int(counts1.get(gene, 0))
            n_mut2 = int(counts2.get(gene, 0))
            n_wt1 = n1 - n_mut1
            n_wt2 = n2 - n_mut2

            freq1 = round(n_mut1 / n1, 4)
            freq2 = round(n_mut2 / n2, 4)

            odds_ratio, p_value = fisher_exact([[n_mut1, n_wt1],
                                                [n_mut2, n_wt2]])
            rows.append({
                "gene": gene,
                f"n_mut_{cohort1_name}": n_mut1,
                f"n_mut_{cohort2_name}": n_mut2,
                f"freq_{cohort1_name}": freq1,
                f"freq_{cohort2_name}": freq2,
                "odds_ratio": round(odds_ratio, 4),
                "p_value": round(p_value, 6),
                "p_value_label": format_pvalue(p_value),
                "significant": p_value < 0.05,
                "enriched_in": cohort1_name if (odds_ratio > 1 and p_value < 0.05)
                else cohort2_name if (odds_ratio < 1 and p_value < 0.05)
                else "none"
            })

        result = pd.DataFrame(rows).sort_values("p_value")

        # plot
        fig, ax = plt.subplots(figsize=(10, 8))
        y = range(len(top_genes))

        freq1_vals = [result[result["gene"] == g][f"freq_{cohort1_name}"].values[0] * 100
                      for g in top_genes]
        freq2_vals = [result[result["gene"] == g][f"freq_{cohort2_name}"].values[0] * 100
                      for g in top_genes]

        ax.barh([i + 0.2 for i in y], freq1_vals, height=0.4,
                color="#2166AC", label=cohort1_name)
        ax.barh([i - 0.2 for i in y], freq2_vals, height=0.4,
                color="#D6604D", label=cohort2_name)

        ax.set_yticks(list(y))
        ax.set_yticklabels(top_genes, fontsize=9)
        ax.set_xlabel("% Samples Mutated", fontsize=12)
        ax.set_title(f"Mutation Frequency: {cohort1_name} vs {cohort2_name}", fontsize=13)
        ax.legend(fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Plot saved to {save_path}")

        plt.show()

        print(f"\nSignificant differences: {result['significant'].sum()}/{len(result)} genes")
        return result.reset_index(drop=True)

    def gene_cooccurrence_with_clinical(self, gene, cohort=None,
                                        oncotree_code=None, cancer_type=None,
                                        center=None, save_path=None):

        from scipy.stats import fisher_exact, mannwhitneyu
        import matplotlib.pyplot as plt

        # get cohort
        if cohort is not None:
            df = cohort.copy()
        else:
            df = self.build_cohort(
                oncotree_code=oncotree_code,
                cancer_type=cancer_type,
                center=center
            )

        df = df.drop_duplicates(subset="PATIENT_ID")

        if len(df) == 0:
            raise ValueError("No patients found")

        # find mutated patients
        mutations = self._load("data_mutations_extended.txt")
        samples = self._load("data_clinical_sample.txt")

        mutated_samples = set(
            mutations[mutations["Hugo_Symbol"].str.upper() == gene.upper()
                      ]["Tumor_Sample_Barcode"]
        )
        sample_to_patient = samples.set_index("SAMPLE_ID")["PATIENT_ID"].to_dict()
        mutated_patients = {sample_to_patient.get(s) for s in mutated_samples}

        df["mutated"] = df["PATIENT_ID"].isin(mutated_patients)
        mut_group = df[df["mutated"] == True]
        wt_group = df[df["mutated"] == False]

        print(f"\n── {gene} Mutation vs Clinical Characteristics ──")
        print(f"   {gene} mutated: {len(mut_group)} patients")
        print(f"   {gene} WT:      {len(wt_group)} patients")

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"{gene} Mutated vs WT — Clinical Characteristics", fontsize=13)

        # Sex
        sex_mut = mut_group["SEX"].value_counts()
        sex_wt = wt_group["SEX"].value_counts()
        sex_df = pd.DataFrame({"Mutated": sex_mut, "WT": sex_wt}).fillna(0)
        sex_df = sex_df[~sex_df.index.isin(["Unknown", "Not Collected", "Not Applicable"])]

        sex_df_pct = sex_df.div(sex_df.sum(axis=0), axis=1) * 100
        sex_df_pct.plot(kind="bar", ax=axes[0], color=["#2166AC", "#D6604D"],
                        rot=45, legend=True)
        axes[0].set_title("Sex", fontsize=11)
        axes[0].set_ylabel("% of group")
        axes[0].spines[["top", "right"]].set_visible(False)

        if "Male" in sex_df.index and "Female" in sex_df.index:
            _, p_sex = fisher_exact([[sex_df.loc["Male", "Mutated"],
                                      sex_df.loc["Female", "Mutated"]],
                                     [sex_df.loc["Male", "WT"],
                                      sex_df.loc["Female", "WT"]]])
            p_label = "< 0.0001" if p_sex < 0.0001 else str(round(p_sex, 4))
            axes[0].set_xlabel(f"p = {p_label}", fontsize=9)

        # Age
        age_col = next((c for c in df.columns if "AGE" in c.upper()), None)
        if age_col:
            mut_ages = pd.to_numeric(mut_group[age_col], errors="coerce").dropna()
            wt_ages = pd.to_numeric(wt_group[age_col], errors="coerce").dropna()

            axes[1].boxplot([mut_ages, wt_ages],
                            labels=[f"{gene} MUT\n(n={len(mut_ages)})",
                                    f"{gene} WT\n(n={len(wt_ages)})"],
                            patch_artist=True,
                            boxprops=dict(facecolor="#2166AC", alpha=0.6))
            axes[1].set_title("Age at Sequencing", fontsize=11)
            axes[1].set_ylabel("Age")
            axes[1].spines[["top", "right"]].set_visible(False)

            _, p_age = mannwhitneyu(mut_ages, wt_ages, alternative="two-sided")
            p_label = "< 0.0001" if p_age < 0.0001 else str(round(p_sex, 4))
            axes[1].set_xlabel(f"p = {p_label}", fontsize=9)

        # Center
        center_col = next((c for c in df.columns if "CENTER" in c.upper()), None)
        if center_col:
            center_mut = mut_group[center_col].value_counts().head(8)
            center_wt = wt_group[center_col].value_counts().head(8)
            center_df = pd.DataFrame({"Mutated": center_mut,
                                      "WT": center_wt}).fillna(0)
            center_df_pct = center_df.div(center_df.sum(axis=0), axis=1) * 100
            center_df_pct.plot(kind="bar", ax=axes[2], color=["#2166AC", "#D6604D"],
                               rot=45, legend=True)
            axes[2].set_title("Sequencing Center", fontsize=11)
            axes[2].set_ylabel("% of group")
            axes[2].spines[["top", "right"]].set_visible(False)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Plot saved to {save_path}")

        plt.show()

        return {
            "mutated": mut_group,
            "wildtype": wt_group,
            "sex_comparison": sex_df,
        }

    def plot_radial_cancer_distribution(self, top_n_cancers=10, top_n_centers=8, save_path=None):
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import numpy as np

        samples = self._load("data_clinical_sample.txt")
        patients = self._load("data_clinical_patient.txt")

        # merge to get center info
        df = samples.merge(patients[["PATIENT_ID", "CENTER"]], on="PATIENT_ID", how="left")
        df = df[df["CANCER_TYPE"] != "UNKNOWN"]

        # get top N cancer types and centers
        top_cancers = df["CANCER_TYPE"].value_counts().head(top_n_cancers).index.tolist()
        top_centers = df["CENTER"].value_counts().head(top_n_centers).index.tolist()

        df = df[df["CANCER_TYPE"].isin(top_cancers) & df["CENTER"].isin(top_centers)]

        # pivot — rows = centers, columns = cancer types
        pivot = df.groupby(["CENTER", "CANCER_TYPE"]).size().unstack(fill_value=0)
        pivot = pivot[top_cancers]  # keep consistent order

        # colors for cancer types
        colors = cm.tab20.colors[:top_n_cancers]

        # setup polar plot
        fig, ax = plt.subplots(figsize=(18, 18), subplot_kw=dict(polar=True))

        n_cancers = len(top_cancers)
        n_centers = len(top_centers)

        # width of each segment
        width = 2 * np.pi / n_cancers

        # ring width
        ring_width = 0.8

        for i, center in enumerate(top_centers):
            if center not in pivot.index:
                continue

            # radius for this ring — innermost = smallest center
            inner_r = 1 + i * ring_width
            outer_r = inner_r + ring_width * 0.85

            row = pivot.loc[center]
            max_val = row.max()

            for j, cancer in enumerate(top_cancers):
                theta = j * width
                value = row[cancer]

                # normalize height within ring
                height = (value / max_val) * ring_width * 0.85 if max_val > 0 else 0

                ax.bar(theta, height, width=width * 0.85,
                       bottom=inner_r, color=colors[j],
                       alpha=0.85, edgecolor="white", linewidth=0.5)

        # add center labels
        for i, center in enumerate(top_centers):
            inner_r = 1 + i * ring_width
            ax.text(0, inner_r + ring_width * 0.4, center,
                    ha="center", va="center", fontsize=8, fontweight="bold")

        # add cancer type labels around the outside
        for j, cancer in enumerate(top_cancers):
            theta = j * width + width
            ax.text(theta, 1 + n_centers * ring_width + 0.2,
                    cancer, ha="center", va="center",
                    fontsize=13, rotation=np.degrees(theta) - 90)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["polar"].set_visible(False)


        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Plot saved to {save_path}")

        plt.show()
