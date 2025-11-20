
import requests
import json

def get_ncbi_summary(protein_id):
    """Fetch general metadata summary from NCBI"""
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=protein&id={protein_id}&retmode=json"
    response = requests.get(url)
    try:
        data = response.json()
        uid = list(data['result'].keys())[1]
        summary = data['result'][uid]
        return {
            "Title": summary.get('title'),
            "Accession": summary.get('accessionversion'),
            "Organism": summary.get('organism'),
            "Length": summary.get('slen'),
            "Source DB": "NCBI Protein",
            "Update Date": summary.get('updatedate'),
            "Definition": summary.get('title'),
            "Tax ID": summary.get('taxid')
        }
    except Exception as e:
        return {"error": f"NCBI summary not available ({e})"}


def get_ncbi_gb_record(protein_id):
    """Fetch GenBank (annotated record) from NCBI"""
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=protein&id={protein_id}&rettype=gb&retmode=text"
    response = requests.get(url)
    if response.ok:
        return response.text
    return "GenBank record not available."


def get_ncbi_fasta(protein_id):
    """Fetch FASTA sequence from NCBI"""
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=protein&id={protein_id}&rettype=fasta&retmode=text"
    response = requests.get(url)
    if response.ok:
        return response.text
    return "FASTA sequence not available."


def get_uniprot_info(protein_id):
    """
    Try fetching UniProt cross-reference from NCBI first,
    or directly use a UniProt ID if known.
    """
    # Try to get a UniProt cross-link using NCBI protein2uniprot mapping
    uniprot_search = f"https://rest.uniprot.org/uniprotkb/search?query={protein_id}&format=json"
    response = requests.get(uniprot_search)
    data = response.json()
    try:
        if data["results"]:
            entry = data["results"][0]
            uid = entry["primaryAccession"]
            detail_url = f"https://rest.uniprot.org/uniprotkb/{uid}.json"
            detail_data = requests.get(detail_url).json()

            protein_info = {
                "UniProt ID": uid,
                "Protein Name": detail_data["proteinDescription"]["recommendedName"]["fullName"]["value"],
                "Organism": detail_data["organism"]["scientificName"],
                "Gene": detail_data["genes"][0]["geneName"]["value"] if detail_data.get("genes") else None,
                "Function": None,
                "Length": detail_data["sequence"]["length"],
                "Sequence": detail_data["sequence"]["value"],
                "GO Terms": [],
                "EC Number": [],
                "Cross-References": [],
            }

            # Extract GO terms (biological process, molecular function, etc.)
            for dbref in detail_data.get("dbReferences", []):
                if dbref.get("type") == "GO":
                    protein_info["GO Terms"].append(f"{dbref.get('id')} - {dbref.get('properties', {}).get('term')}")

            # Extract EC number if available
            if "recommendedName" in detail_data["proteinDescription"]:
                names = detail_data["proteinDescription"]["recommendedName"]
                if "ecNumbers" in names:
                    protein_info["EC Number"] = [ec["value"] for ec in names["ecNumbers"]]

            # Extract function text if available
            comments = detail_data.get("comments", [])
            for c in comments:
                if c.get("commentType") == "FUNCTION":
                    protein_info["Function"] = c["texts"][0]["value"]

            # Extract cross-references (PDB, EMBL, etc.)
            for dbref in detail_data.get("uniProtKBCrossReferences", []):
                if dbref["database"] in ["PDB", "EMBL", "RefSeq", "InterPro"]:
                    protein_info["Cross-References"].append({
                        "Database": dbref["database"],
                        "ID": dbref.get("id")
                    })

            return protein_info
    except Exception as e:
        return {"error": f"UniProt data not available ({e})"}

    return {"error": "No UniProt record found for this ID."}


def get_full_protein_info(protein_id):
    """Combine all protein details from multiple sources"""
    info = {
        "NCBI Summary": get_ncbi_summary(protein_id),
        "NCBI FASTA": get_ncbi_fasta(protein_id),
        "NCBI GenBank Record": get_ncbi_gb_record(protein_id),
        "UniProt Data": get_uniprot_info(protein_id)
    }
    return info


# ---------------- RUN EXAMPLE ----------------
protein_id = "CAM80794.1"  # Example from your professor
info = get_full_protein_info(protein_id)

# Print prettified JSON
print(json.dumps(info, indent=4))



