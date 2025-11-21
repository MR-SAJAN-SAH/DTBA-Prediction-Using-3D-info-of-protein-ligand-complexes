#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from rdkit import Chem
from rdkit.Chem import AllChem
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def generate_ligand_3d(smiles: str):
    """Generate 3D coordinates for ligand from SMILES"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    
    # Add hydrogens and generate 3D coordinates
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)
    
    # Get coordinates and atom types
    conf = mol.GetConformer()
    coords = []
    atom_types = []
    colors = []
    atom_sizes = []
    
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        coords.append([pos.x, pos.y, pos.z])
        atom_symbol = atom.GetSymbol()
        atom_types.append(atom_symbol)
        
        # Color coding by atom type
        color_map = {'C': 'black', 'O': 'red', 'N': 'blue', 'S': 'yellow', 'P': 'orange', 'H': 'white'}
        colors.append(color_map.get(atom_symbol, 'gray'))
        
        # Size by atom type
        size_map = {'C': 40, 'O': 45, 'N': 45, 'S': 50, 'P': 50, 'H': 20}
        atom_sizes.append(size_map.get(atom_symbol, 40))
    
    # Get bonds
    bonds = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        bonds.append((a1, a2))
    
    return np.array(coords), atom_types, colors, atom_sizes, bonds

def generate_protein_3d(sequence: str):
    """Generate approximate 3D structure for protein from sequence"""
    n_residues = len(sequence)
    
    # Create alpha-helix like backbone (simplified)
    ca_coords = []
    residue_types = []
    
    for i, aa in enumerate(sequence):
        # Simple helix parameters
        radius = 5.0
        rise_per_residue = 1.5
        degrees_per_residue = 100
        
        theta = i * degrees_per_residue * (np.pi/180)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        z = i * rise_per_residue
        ca_coords.append([x, y, z])
        residue_types.append(aa)
    
    ca_coords = np.array(ca_coords)
    
    # Color by residue type
    residue_colors = {
        'A': 'red', 'R': 'blue', 'N': 'green', 'D': 'green', 'C': 'yellow',
        'Q': 'green', 'E': 'green', 'G': 'orange', 'H': 'blue', 'I': 'red',
        'L': 'red', 'K': 'blue', 'M': 'red', 'F': 'purple', 'P': 'orange',
        'S': 'green', 'T': 'green', 'W': 'purple', 'Y': 'purple', 'V': 'red'
    }
    
    colors = [residue_colors.get(aa, 'gray') for aa in residue_types]
    
    return ca_coords, residue_types, colors

def visualize_complex_interactive(drug_id, smiles, protein_sequence):
    """Interactive 3D visualization using Plotly"""
    
    # Generate ligand 3D
    lig_coords, lig_atoms, lig_colors, lig_sizes, lig_bonds = generate_ligand_3d(smiles)
    
    # Generate protein 3D
    prot_coords, prot_residues, prot_colors = generate_protein_3d(protein_sequence)
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        subplot_titles=('Ligand Structure', 'Protein Structure')
    )
    
    # Plot ligand
    fig.add_trace(
        go.Scatter3d(
            x=lig_coords[:, 0], y=lig_coords[:, 1], z=lig_coords[:, 2],
            mode='markers',
            marker=dict(
                size=lig_sizes,
                color=lig_colors,
                opacity=0.8
            ),
            text=lig_atoms,
            name='Ligand Atoms'
        ),
        row=1, col=1
    )
    
    # Add ligand bonds
    for bond in lig_bonds:
        a1, a2 = bond
        fig.add_trace(
            go.Scatter3d(
                x=[lig_coords[a1, 0], lig_coords[a2, 0]],
                y=[lig_coords[a1, 1], lig_coords[a2, 1]],
                z=[lig_coords[a1, 2], lig_coords[a2, 2]],
                mode='lines',
                line=dict(color='black', width=5),
                showlegend=False
            ),
            row=1, col=1
        )
    
    # Plot protein backbone
    fig.add_trace(
        go.Scatter3d(
            x=prot_coords[:, 0], y=prot_coords[:, 1], z=prot_coords[:, 2],
            mode='markers+lines',
            marker=dict(
                size=6,
                color=prot_colors,
                opacity=0.8
            ),
            line=dict(color='gray', width=4),
            text=prot_residues,
            name='Protein Backbone'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title=f"3D Structure Visualization - Drug {drug_id}",
        height=600,
        showlegend=True
    )
    
    fig.show()

def visualize_complex_static(drug_id, smiles, protein_sequence):
    """Static 3D visualization using Matplotlib"""
    
    # Generate structures
    lig_coords, lig_atoms, lig_colors, lig_sizes, lig_bonds = generate_ligand_3d(smiles)
    prot_coords, prot_residues, prot_colors = generate_protein_3d(protein_sequence)
    
    fig = plt.figure(figsize=(15, 6))
    
    # Ligand subplot
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Plot ligand atoms
    scatter = ax1.scatter(lig_coords[:, 0], lig_coords[:, 1], lig_coords[:, 2], 
                         c=lig_colors, s=lig_sizes, alpha=0.8)
    
    # Plot ligand bonds
    for bond in lig_bonds:
        a1, a2 = bond
        ax1.plot([lig_coords[a1, 0], lig_coords[a2, 0]],
                [lig_coords[a1, 1], lig_coords[a2, 1]],
                [lig_coords[a1, 2], lig_coords[a2, 2]], 'k-', linewidth=2)
    
    ax1.set_title(f'Ligand: {drug_id}\nSMILES: {smiles}')
    ax1.set_xlabel('X (Å)')
    ax1.set_ylabel('Y (Å)')
    ax1.set_zlabel('Z (Å)')
    
    # Protein subplot
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Plot protein backbone
    for i in range(len(prot_coords)-1):
        ax2.plot([prot_coords[i, 0], prot_coords[i+1, 0]],
                [prot_coords[i, 1], prot_coords[i+1, 1]],
                [prot_coords[i, 2], prot_coords[i+1, 2]], 
                color='gray', linewidth=2)
    
    scatter = ax2.scatter(prot_coords[:, 0], prot_coords[:, 1], prot_coords[:, 2],
                         c=prot_colors, s=50, alpha=0.8)
    
    ax2.set_title(f'Protein: {len(protein_sequence)} residues')
    ax2.set_xlabel('X (Å)')
    ax2.set_ylabel('Y (Å)')
    ax2.set_zlabel('Z (Å)')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"Drug ID: {drug_id}")
    print(f"Ligand: {len(lig_atoms)} atoms")
    print(f"Protein: {len(protein_sequence)} residues")
    print(f"SMILES: {smiles}")
    print(f"Protein sequence (first 50): {protein_sequence[:50]}...")

# Test with your data
if __name__ == "__main__":
    example_data = {
        'drug_id': ['65163'],
        'smiles': ['C1=C(C(=O)OC1=O)CC(=O)O'],
        'protein_id': ['WP_146695988.1'],
        'protein_sequence': ['MKFTSYNIAALWSYIKKGDIRAIAVYGNDHGLIDYRCQEITKLFNANLRIYDYRELTEADFIFILNSNNLFSQREIVKIYNTPGNINAALKKALTFNNQNFLIVLGNEFSASSTTRQWFETQKYLAALGCYTENSQDIKKLLSQIVNKAGKNITAEAAVYFSNTAYGDKYCYINEINKLILYCHDCDTISKKEVNKCISTEILGTSDLMCIYFAKGIAMNFFKEVEKIRNNNIPDVWILRALIRYYINLYIVLMKREHGVSIEQAIKSIQPSIFFKYVQDFQLIAQTKTLNNVLHTLNELYMAELSVKTTHHHINNIIEVVFLKITHKLRLSFEFTVLV'],
        'affinity_value': ['3.73']
    }
    
    drug_id = example_data['drug_id'][0]
    smiles = example_data['smiles'][0]
    protein_sequence = example_data['protein_sequence'][0]
    
    print("Generating 3D structures...")
    
    # Interactive visualization (opens in browser)
    visualize_complex_interactive(drug_id, smiles, protein_sequence)
    
    # Static visualization
    visualize_complex_static(drug_id, smiles, protein_sequence)