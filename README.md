To use:

Both Phase 1 and Phase 2 work the same.

- First run the tuner file (either BioBERT_Tuner.py or BioBERT_NER_Tuner.py) to build the model.
- Then run the tester file (either BioBERT_Tester.py or BioBERT_NER_Tester.py) to test the model.

The tuner file creates a a directory that contains the fine-tuned model. Then, the tester file takes the fine-tuned model created by the tuner and evaluates it compared to an untuned BioBERT model.
If you are getting an error while running the tester file, delete the directories created by the tuner files (tmp_trainer and biobert-sentence-model for Phase 1, and tmp_trainer, biobert-ner, and biobert-ner-model for Phase 2) then rerun the tuner file. You may need to close your IDE to delete these files.
