import pandas as pd
import random
import os

side_effects = [
    "nausea","headache","dizziness","fatigue","rash",
    "vomiting","insomnia","anxiety","fever","dry_mouth"
]

drug_types = ["antibiotic","painkiller","antidepressant","antiviral","steroid"]
conditions = ["diabetes","hypertension","asthma","infection","arthritis"]

data = []

for _ in range(3000):
    description = f"{random.choice(drug_types)} used for {random.choice(conditions)} treatment"
    age = random.randint(18,80)
    gender = random.randint(0,1)
    dosage = random.randint(100,1000)

    effects = [random.randint(0,1) for _ in range(10)]

    row = [description, age, gender, dosage] + effects
    data.append(row)

columns = ["description","age","gender","dosage"] + side_effects

df = pd.DataFrame(data, columns=columns)

os.makedirs("dataset", exist_ok=True)
df.to_csv("dataset/drug_data.csv", index=False)

print("✅ Dataset Generated Successfully!")
