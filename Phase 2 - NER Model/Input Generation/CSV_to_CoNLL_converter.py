"""
This script takes converts the input CSV file from Phase 1 into a CoNLL format for used for training BioBERT_NER
"""

import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher
from spacy.language import Language
from spacy.tokens import Span
import sys

# Reading input from command line args
input_path = sys.argv[1]
output_path = sys.argv[2]

# Creating custom model (using models existing pipelines, minus NER)
nlp = spacy.load('en_core_web_sm')

# Add a phrase matcher
matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

# Add the entity ruler component to the pipeline
ruler = nlp.add_pipe("entity_ruler", config={"overwrite_ents": True})

# List of genomic patterns to match
genomic_terms = [
                "raw methylation", "methylation values", "raw-methylation", "sequencing data", "RRBS",  "methylation profiling",
                "copy number variations", "WES", "whole exome sequence", "whole genome sequence", "WGS", "DNA sequence",
                "transcriptome sequencing", "expression data", "expression profiling", "mRNA expression", "RNA sequence", "RNA samples",
                "microarray data"
                ]
genomic_patterns = [nlp.make_doc(text) for text in genomic_terms]
matcher.add("GENOMIC_DATA_TYPE", genomic_patterns)

# List of cancer type strings to match
cancer_types = [
    "ACC", "Adrenocortical carcinoma", "BLCA", "Bladder Urothelial carcinoma", "BRC", "BRCA", "Breast invasive carcinoma", "CESC", "Cervical squamous cell carcinoma and endocervical adenocarcinoma", "CHOL", "Cholangiocarcinoma", "COAD", "Colon adenocarcinoma", "DLBC", "Lymphoid Neoplasm Diffuse Large B-cell Lymphoma", "lymphoma", "ESCA", "Esophageal carcinoma", "GBM", "Glioblastoma multiforme", "HNSC", "Head and Neck squamous cell carcinoma", "KICH", "Kidney Chromophobe", "KIRC", "Kidney renal clear cell carcinoma", "KIRP", "Kidney renal papillary cell carcinoma", "LAML", "Acute Myeloid Leukemia", "LGG", "Brain Lower Grade Glioma", "LIHC", "Liver hepatocellular carcinoma", "LUAD", "Lung adenocarcinoma", "LUSC", "Lung squamous cell carcinoma", "MESO", "Mesothelioma", "OV", "Ovarian serous cystadenocarcinoma", "PAAD", "Pancreatic adenocarcinoma", "PCPG", "Pheochromocytoma and Paraganglioma", "PRAD", "Prostate adenocarcinoma", "READ", "Rectum adenocarcinoma", "SARC", "Sarcoma", "SKCM", "Skin Cutaneous Melanoma", "STAD", "Stomach adenocarcinoma", "TGCT", "Testicular Germ Cell Tumors", "THCA", "Thyroid carcinoma", "THYM", "Thymoma", "UCEC", "Uterine Corpus Endometrial Carcinoma", "UCS", "Uterine Carcinosarcoma", "UVM", "Uveal Melanoma", "breast", "lung", "kidney", "renal", "bladder", "brain", "liver", "prostate" 
]
cancer_patterns = [nlp.make_doc(text) for text in cancer_types]
matcher.add("CANCER_TYPE", cancer_patterns)

# Add the matcher to the pipeline
@Language.component("add_matcher_entities")
def add_matcher_entities(doc):
    matches = matcher(doc)
    spans = []
    
    # Create spans from matcher results
    for match_id, start, end in matches:
        label = nlp.vocab.strings[match_id]
        spans.append(Span(doc, start, end, label=label))
    
    # Get existing entities 
    existing_ents = list(doc.ents)
    all_spans = spans + existing_ents
    
    # Define priority order
    priorities = {
        "SAMPLE_COUNT": 10,
        "GENOMIC_DATA_TYPE": 8,
        "DATA_SOURCE": 6,
        "DATA_ACCESSION": 4,
        "CANCER_TYPE": 2
    }
    
    # Sort spans by priority, then by length (longest first)
    sorted_spans = sorted(
        all_spans, 
        key=lambda span: (priorities.get(span.label_, 0), len(span)),
        reverse=True
    )
    
    # Filter spans to remove overlaps, keeping highest priority spans
    doc.ents = filter_spans(sorted_spans)
    return doc

def filter_spans(spans):
    sorted_spans = sorted(spans, key=lambda span: (span.end - span.start, -span.start))
    result = []
    seen_tokens = set()
    for span in sorted_spans:
        # Check for overlap
        if any(token.i in seen_tokens for token in span):
            continue
        result.append(span)
        seen_tokens.update(token.i for token in span)
    return result

nlp.add_pipe("add_matcher_entities", after="entity_ruler")

# Define patterns for custom entities
patterns2 = [
    {"label": "DATA_SOURCE", "pattern": "OncoSG"},
    {"label": "DATA_SOURCE", "pattern": "dbGaP"},
    {"label": "DATA_SOURCE", "pattern": "SRA"},
    {"label": "DATA_SOURCE", "pattern": "EGA"},
    {"label": "DATA_SOURCE", "pattern": "TCGA"},
    {"label": "DATA_SOURCE", "pattern": "CPTAC"},
    {"label": "DATA_SOURCE", "pattern": "UCSC"},
    {"label": "DATA_SOURCE", "pattern": [{"LOWER": "european"}, {"LOWER": "genome"}, {"OP": "?", "LOWER": "-"}, 
                                        {"LOWER": "phenome"}, {"LOWER": "archive"}]},
    {"label": "DATA_SOURCE", "pattern": [{"LOWER": "gene"}, {"LOWER": "expression"}, {"LOWER": "omnibus"}]},
    {"label": "DATA_SOURCE", "pattern": [{"LOWER": "sequence"}, {"LOWER": "read"}, {"LOWER": "archive"}]},
    {"label": "DATA_SOURCE", "pattern": [{"LOWER": "database"}, {"LOWER": "of"}, {"LOWER": "genotypes"}, {"LOWER": "and"}, {"LOWER": "phenotypes"}]},
    {"label": "DATA_ACCESSION", "pattern": [{"TEXT": {"REGEX": "EGA[S,D][0-9]{11}"}}]},
    {"label": "DATA_ACCESSION", "pattern": [{"TEXT": {"REGEX": "GSE[0-9]{1,10}"}}]},
    {"label": "SAMPLE_COUNT", 
    "pattern": [
        {"IS_DIGIT": True}, 
        {"OP": "{0,3}"},     
        {"LEMMA": {"IN": ["patient", "sample", "tumor", "tissue"]}} 
    ]},
    {"label": "DATA_ACCESSION", 
    "pattern": [
        {"LEMMA": {"IN": ["host", "available"]}}, 
        {"OP": "{0,2}"},     
        {"LIKE_URL": True} 
    ]},
]

# Add patterns to ruler
ruler.add_patterns(patterns2)

# Reads text from rows and writes to file in CoNLL format
def process_text(text, extra_input, filepath):
    file = open(filepath, "a")

    doc = nlp(text)

    # Print entities and CoNLL format
    # print("Entities found:")
    # for ent in doc.ents:
    #     print(f"{ent.text} - {ent.label_}")

    # print("\nCoNLL format:")
    for token in doc:
        # Get entity tag (BIO format)
        if token.ent_type_:
            if token.ent_iob_ == "B":
                ent_tag = f"B-{token.ent_type_}"
            else:
                ent_tag = f"I-{token.ent_type_}"
        else:
            ent_tag = "O"
            
        file.write(f"{token.text}\t{token.pos_}\t{token.tag_}\t{token.dep_}\t{ent_tag}\t{extra_input}")
        file.write("\n")
    file.write("\n")
    file.close()

input_df = pd.read_csv(input_path, index_col = 0)
input_df = input_df.dropna()
input_df = input_df.query('processed == 0')

sources = input_df.groupby('source')

for s in sources.groups.keys():
    tdf = sources.get_group(s)
    for index, row in tdf.iterrows():
        process_text(row.text, s, output_path)