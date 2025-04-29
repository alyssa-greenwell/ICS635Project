This script converts the input used in Phase 1 to CoNLL input format used in Phase 2. It tokenizes the sentences in Phase 1's input and annotates them with BIO format for 5 custom entities. For full annotation of entities as labeled by SpaCy's `en_core_web_sm` model + 5 custom entities, checkout `full_NER` branch.

To use:
Run with:
`python CSV_to_CoNLL_converter.py {path/to/input.csv} {path/to/output.txt}`
The script expects a CSV input in the same format as phase1_sentences.csv in [Phase 1](../../Phase 1 - Binary Classifier). 
Example input is in [input/example_input.csv](input/example_input.csv).
Example output is in [output/example_output.txt](output/example_output.txt)