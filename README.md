# Hybrid_ML_Fault_Detection

This repository accompanies a conference paper on a hybrid machine-learning
framework for predictive fault detection in cloud computing environments.
The experiment uses a predictive-maintenance dataset with machine operating
measurements, rather than cloud telemetry.

## Experiment

The notebook `hybrid_final_corrected.ipynb` and `predictive_maintenance.csv`
are available on the repository's `main` branch. The notebook predicts the
binary `Target` column after dropping identifiers and `Failure Type` and
one-hot encoding `Type`. It uses a stratified 80/20 train/test split,
standardizes features using the training split, and applies SMOTE to the
training data.

The hybrid model stacks Random Forest, XGBoost, and a feed-forward neural
network. A logistic regression meta-model learns from five-fold out-of-fold
base-model probabilities. Scaling and SMOTE are performed before these inner
folds, so the inner-fold predictions are not independent of preprocessing.

The notebook compares the hybrid with its three base models using accuracy,
precision, recall, F1, ROC-AUC, confusion matrices, and plots. It also runs
stratified five-fold cross-validation with scaling and SMOTE within each outer
training fold, followed by paired statistical comparisons. Results are
exported as Excel files.

## Running the notebook

Open the notebook from a revision containing the notebook and CSV in Google
Colab or Jupyter and run its cells in order. The first cell installs the
required Python packages. In Colab, upload `predictive_maintenance.csv` when
prompted; locally, place it in the notebook's working directory. The notebook
notes that the experiment takes roughly 10–15 minutes on a Colab CPU.
