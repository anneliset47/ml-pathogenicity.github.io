import pandas as pd

# Load Excel file
df = pd.read_excel('/Users/annelisethorn/Documents/GitHub/tr-plots/Code/Matching Files/Excels/83_loci_503_samples_withancestrycolumns.xlsx')

# Load sample info with gender
sample_info = pd.read_csv('/Users/annelisethorn/Documents/GitHub/tr-plots/20130606_sample_info.txt', sep='\t', usecols=['Sample', 'Gender'])

# Prepare for merging
sample_info = sample_info.rename(columns={'Sample': 'Base Sample ID', 'Gender': 'Sex'})
df['Base Sample ID'] = df['Sample ID'].apply(lambda x: x.split('-')[0] if isinstance(x, str) else x)

# Merge sex info
df = df.merge(sample_info, on='Base Sample ID', how='left')

# if sex is blank for a row, fill with 'Unknown'
df['Sex'] = df['Sex'].fillna('Unknown')

# make sure sex entries are title case
df['Sex'] = df['Sex'].str.title()

# (Optional) Save to Excel to inspect
df.to_excel('/Users/annelisethorn/Documents/GitHub/tr-plots/83_loci_503_samples_with_sex4.xlsx', index=False)

# Print merge complete message
print("Merge complete")