
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Set style
sns.set(style="whitegrid")
plt.rcParams.update({'figure.figsize': (10, 6)})

# Load datasets
clinvar_df = pd.read_csv("clinvar_repeat_pathogenic_variants.csv")
ensembl_df = pd.read_csv("ensembl_tandem_repeats.csv")
populations_df = pd.read_csv("igsr-1kg_ont_vienna-populations.tsv", sep="\t")
samples_df = pd.read_csv("igsr-1kg_ont_vienna-samples.tsv", sep="\t")

# 1. Clinical Significance Distribution
sns.countplot(data=clinvar_df, y="ClinicalSignificance",
              order=clinvar_df["ClinicalSignificance"].value_counts().index)
plt.title("Clinical Significance Distribution")
plt.xlabel("Count")
plt.tight_layout()
plt.savefig("1_clinical_significance_distribution.png")
plt.clf()

# 2. Top 10 Genes by Variant Count
top_genes = clinvar_df['Gene'].value_counts().nlargest(10)
sns.barplot(x=top_genes.values, y=top_genes.index)
plt.title("Top 10 Genes by Variant Count")
plt.xlabel("Variant Count")
plt.tight_layout()
plt.savefig("2_top_10_genes.png")
plt.clf()

# 3. Word Cloud of Phenotypes
phenotype_text = " ".join(clinvar_df['Phenotypes'].dropna().astype(str))
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(phenotype_text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Word Cloud of Phenotypes")
plt.tight_layout()
plt.savefig("3_wordcloud_phenotypes.png")
plt.clf()

# 4. Number of Phenotypes per Variant
phenotype_counts = clinvar_df['Phenotypes'].apply(lambda x: len(str(x).split(";")) if pd.notnull(x) else 0)
sns.histplot(phenotype_counts, bins=10)
plt.title("Number of Phenotypes per Variant")
plt.xlabel("Phenotype Count")
plt.ylabel("Variant Frequency")
plt.tight_layout()
plt.savefig("4_phenotypes_per_variant.png")
plt.clf()

# 5. Tandem Repeat Length Distribution
ensembl_df['length'] = ensembl_df['end'] - ensembl_df['start']
sns.histplot(ensembl_df['length'], bins=50)
plt.title("Tandem Repeat Length Distribution")
plt.xlabel("Length (bp)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("5_repeat_length_distribution.png")
plt.clf()

# 6. Repeats by Chromosome
chrom_counts = ensembl_df['seq_region_name'].value_counts().head(20)
sns.barplot(x=chrom_counts.index, y=chrom_counts.values)
plt.title("Repeats by Chromosome")
plt.xlabel("Chromosome")
plt.ylabel("Repeat Count")
plt.tight_layout()
plt.savefig("6_repeats_by_chromosome.png")
plt.clf()

# 7. Population by Superpopulation
superpop_counts = populations_df['Superpopulation name'].value_counts()
sns.barplot(x=superpop_counts.values, y=superpop_counts.index)
plt.title("Population by Superpopulation")
plt.xlabel("Count")
plt.ylabel("Superpopulation")
plt.tight_layout()
plt.savefig("7_population_superpopulation.png")
plt.clf()

# 8. Sample Sex Distribution
sex_counts = samples_df['Sex'].value_counts()
sns.barplot(x=sex_counts.index, y=sex_counts.values)
plt.title("Sample Sex Distribution")
plt.xlabel("Sex")
plt.ylabel("Sample Count")
plt.tight_layout()
plt.savefig("8_sample_sex_distribution.png")
plt.clf()

# 9. Top 10 Populations by Sample Count
pop_sample_counts = samples_df['Population name'].value_counts().nlargest(10)
sns.barplot(x=pop_sample_counts.values, y=pop_sample_counts.index)
plt.title("Top 10 Populations by Sample Count")
plt.xlabel("Sample Count")
plt.ylabel("Population")
plt.tight_layout()
plt.savefig("9_top_populations_samples.png")
plt.clf()

# 10. Phenotype Count by Clinical Significance
clinvar_df['PhenotypeCount'] = phenotype_counts
sns.boxplot(data=clinvar_df, x="ClinicalSignificance", y="PhenotypeCount")
plt.title("Phenotype Count by Clinical Significance")
plt.xlabel("Clinical Significance")
plt.ylabel("Number of Phenotypes")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("10_phenotypes_by_significance.png")
plt.clf()
