import requests
import pandas as pd

# Get repeat features on chromosome 1:1-1000000
url = "https://rest.ensembl.org/overlap/region/human/1:1-1000000?feature=repeat"
headers = { "Content-Type" : "application/json" }

response = requests.get(url, headers=headers)

if not response.ok:
    print(response.text)
else:
    repeats = response.json()

# Save results to CSV
df = pd.DataFrame(repeats)
df.to_csv("ensembl_tandem_repeats.csv", index=False)
print("Saved to 'ensembl_tandem_repeats.csv'")
