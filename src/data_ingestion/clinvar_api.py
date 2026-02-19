from Bio import Entrez
import pandas as pd
import time
import pprint

# Set your email for NCBI Entrez
Entrez.email = "anth6800@colorado.edu"

# Search for repeat expansions with pathogenic classification in ClinVar
search_term = 'repeat expansion AND pathogenic[ClinicalSignificance]'
handle = Entrez.esearch(db="clinvar", term=search_term, retmax=100)
record = Entrez.read(handle)
handle.close()

# Extract variant IDs from the search results
variant_ids = record['IdList']
print(f"Found {len(variant_ids)} variants")

# Fetch detailed information for each variant in chunks
variant_data = []

# Iterate over variant IDs in chunks to avoid hitting API limits
for i in range(0, len(variant_ids), 10):
    id_chunk = variant_ids[i:i+10]
    ids = ",".join(id_chunk)
    
    # Fetch summaries for the current chunk
    handle = Entrez.esummary(db="clinvar", id=ids, retmode="xml")
    records = Entrez.read(handle, validate=False)
    handle.close()

    # Process each record
    for rec in records['DocumentSummarySet']['DocumentSummary']:
        variation_id = rec.attributes.get('uid', 'N/A')
        accession = rec.get('accession', 'N/A')
        title = rec.get('title', '')
        gene = rec.get('gene_sort', '')
        clinical_significance = rec.get('germline_classification', {}).get('description', '')
        
        trait_set = rec.get('germline_classification', {}).get('trait_set', [])
        phenotypes = "; ".join([t.get('trait_name', '') for t in trait_set]) if trait_set else ''

        variant_data.append({
            "VariationID": variation_id,
            "Accession": accession,
            "Gene": gene,
            "Title": title,
            "ClinicalSignificance": clinical_significance,
            "Phenotypes": phenotypes
        })

    time.sleep(0.3)

# Convert the collected data to a DataFrame and save to CSV
df = pd.DataFrame(variant_data)
df.to_csv("clinvar_repeat_pathogenic_variants.csv", index=False)
print("Saved to 'clinvar_repeat_pathogenic_variants.csv'")
