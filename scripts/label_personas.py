import json
import random

# Re-label the existing dataset with persona orientation.
# This does NOT require re-downloading; it works on the already-fetched sycophancy_dataset.json.

def classify_persona(prompt):
    p = prompt.lower()
    if 'liberal' in p or 'progressive' in p or 'democrat' in p:
        return 'liberal'
    elif 'conservative' in p or 'republican' in p or 'right-wing' in p or 'traditionalist' in p:
        return 'conservative'
    return 'unknown'

with open('sycophancy_dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

liberal = [d for d in data if classify_persona(d['prompt']) == 'liberal']
conservative = [d for d in data if classify_persona(d['prompt']) == 'conservative']

print(f"Dataset persona breakdown: liberal={len(liberal)}, conservative={len(conservative)}")

# Add the persona_orientation field to each item
for item in data:
    item['persona_orientation'] = classify_persona(item['prompt'])

# Save back with labels
with open('sycophancy_dataset.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"Saved updated dataset with persona_orientation labels.")
