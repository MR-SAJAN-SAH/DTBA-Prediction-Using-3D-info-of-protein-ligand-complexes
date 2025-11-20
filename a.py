'''import torch

# ✅ Force full pickle loading (since it's not just weights)
data = torch.load("processed_dataset/1a1c.pt", map_location='cpu', weights_only=False)

print(data.keys())
print("Binding affinity (pKd):", data["y"].item())
print("Ligand nodes:", data["ligand"].x.shape)
print("Protein residues:", len(data["residue_names"]))
print("Cross edges:", data["cross_edges"].shape)
'''

'''
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors

smiles = "C1=C(C(=O)OC1=O)CC(=O)O"
mol = Chem.MolFromSmiles(smiles)

# Try to generate IUPAC name (needs RDKit with name generation support)
try:
    from rdkit.Chem import rdMolDescriptors
    name = rdMolDescriptors.CalcMolFormula(mol)
    print("Molecular Formula:", name)
except:
    print("IUPAC name generation not supported in this RDKit build.")

# You can also print general info
print("Molecular Weight:", Descriptors.MolWt(mol))

'''

'''

import requests

smiles = "C1=C(C(=O)OC1=O)CC(=O)O"
url = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/"
    f"{smiles}/property/"
    "MolecularFormula,MolecularWeight,CanonicalSMILES,IsomericSMILES,"
    "InChI,InChIKey,IUPACName,Title,ExactMass,MonoisotopicMass,"
    "TPSA,Complexity,Charge,HBondDonorCount,HBondAcceptorCount,"
    "RotatableBondCount,XLogP/JSON"
)

response = requests.get(url)
data = response.json()

try:
    props = data['PropertyTable']['Properties'][0]
    for key, value in props.items():
        print(f"{key}: {value}")
except Exception as e:
    print("Compound not found or error:", e)
'''


import requests

protein_id = "CAM80794.1"
url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=protein&id={protein_id}&retmode=json"

response = requests.get(url)
data = response.json()
print(data)
