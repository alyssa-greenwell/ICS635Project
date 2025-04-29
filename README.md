To use:

Both Phase 1 and Phase 2 work the same.

- First run the tuner file (either BioBERT_Tuner.py or BioBERT_NER_Tuner.py) to build the model.
- Then run the tester file (either BioBERT_Tester.py or BioBERT_NER_Tester.py) to test the model.

The tuner file creates a a directory that contains the fine-tuned model. Then, the tester file takes the fine-tuned model and evaluates it compared to an untuned BioBERT model.
If you are getting an error while running the tester file, delete the biobert-sentence-model directory and rerun the tuner file. You may need to close your IDE to do so.
