# Retraining Instructions

1) Create a working directory.
2) Copy J_Medline.txt, lsi2018.xml, and selectivly-indexed-journals-of-interest.csv from ./input_data to the working directory.
3) In the working directory create a folder called medline_data.
4) Download the required daily update files 929 - 1250 from https://github.com/indexing-initiative/selective_indexing/releases
5) Unzip the files into the medline_data folder.
6) Create an anconda environment:

```
conda create --name selective_indexing --file requirements.txt
conda activate selective_indexing
conda install requests
pip install tensorflow-gpu==2.0.1
```

7) Download the data for the tokenizer:

```
python -m nltk.downloader punkt
```

9) Run the retraining script:

```
python -m BmCS.retrain --workdir /path/to/workdir
```

10) When the script has finished, copy the following files to the BmCS/models folder in the biomedical-citation-selector repository:
      - journal_ids.txt
      - word_indices.txt

11) Note that the retrained models have been saved to the follwoing folders in the working directory:
      - ./cnn_model/checkpoints/best_model.hdf5 (new CNN model)
      - ./voting_model/voting_model.joblib (new ensemble model)

12) Note that optimized threshold values have been saved in the following files in the working directory:
      - combined_optimum_thresholds.txt (thresholds to use for combined model)
      - voting_optimum_thresholds.txt (thresholds to use for ensemble model)
      - cnn_optimum_thresholds.txt (thresholds to use for CNN model)

13) In BmCS/thresholds.py in the biomedical-citation-selector repository, update COMBINED_THRESH with the high recall threshold saved in the first line of the combined_optimum_thresholds.txt file, and also update PRECISION_THRESH with the high precision theshold saved in the second line of the combined_optimum_thresholds.txt file.

14) (optional) In BmCS/thresholds.py update  VOTING_THRESH and CNN_THRESH with the high recall thresholds saved in the first lines of the voting_optimum_thresholds.txt and cnn_optimum_thresholds.txt files respecitively. These threshold values are only used by the --validation and --test BmCS command line options. To get the --validation and --test assertions to pass the tolerances in the BmCS/BmCS_tests/BmCS_test.py file will need to be increased.
