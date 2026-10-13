# Clinical Trial Keyword Recommendation Using Collaborative Filtering

## 1. Project Overview

This project implements a collaborative filtering-based recommendation system that recommends relevant Medical Subject Headings (MeSH) keywords for clinical trials.

The system learns latent relationships between clinical trials and MeSH terms to predict suitable keywords for a given trial. The implementation is based on the original research code and includes memory optimizations to support training on systems with limited RAM.

**Original reference implementation:**  
https://github.com/xiaomutt/xliniq

**Project repository:**  
https://github.com/nehaha21/clinical-trial-keyword-recommendation

## 2. Objectives

- Recommend relevant MeSH keywords for clinical trials.
- Implement a Mini-Batch ReLU Latent Factor Model.
- Evaluate recommendation performance using the project's existing metrics.
- Reduce memory consumption during training and evaluation.
- Enable model training on systems with limited RAM.

## 3. Dataset

The project uses the ClinicalTrials.gov dataset with MeSH keyword annotations.

Expected dataset path:

```text
output/
└── ctg-public-xml.sds
```

Dataset details used in the current experiments:

| Property | Value |
|---|---:|
| Total clinical trials | 606,387 |
| MeSH vocabulary size | 16,089 |
| Training samples | 485,074 |
| Validation samples | 60,295 |
| Test samples | 61,018 |

**Note:** The dataset is not included in the repository. Obtain the dataset from the appropriate source and place it at the expected path before running the scripts.

## 4. Model Description

The project uses a Mini-Batch ReLU Latent Factor Model to predict trial–MeSH associations.

The prediction formulation is:

\[
\hat{B} = \operatorname{ReLU}((W \odot \Theta)B+b)
\]

where:

- \(B\) represents the trial–keyword interaction data.
- \(W\) and \(\Theta\) represent learned model parameters.
- \(\odot\) denotes element-wise multiplication.
- \(b\) represents the bias term.
- ReLU introduces non-linearity.
- \(\hat{B}\) represents predicted trial–keyword associations.

The model is trained using mini-batches and evaluated using the metrics implemented in the original project.

## 5. Requirements

- Python 3.11
- Git
- Sufficient RAM and disk space for the dataset and model checkpoints

Install the Python dependencies listed in `requirements.txt`.

## 6. Installation and Setup

### Step 1: Clone the repository

```bash
git clone https://github.com/nehaha21/clinical-trial-keyword-recommendation.git
cd clinical-trial-keyword-recommendation
```

### Step 2: Create a virtual environment

**Windows:**

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
```

**Linux/macOS:**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If the repository does not contain `requirements.txt`, install the dependencies specified by the project and its original implementation before proceeding.

### Step 4: Prepare the dataset

Create the output directory if it does not exist:

```bash
mkdir output
```

Place the dataset at:

```text
output/ctg-public-xml.sds
```

Ensure that the dataset format matches what the project's preprocessing and training scripts expect.

## 7. Usage

### Train the model

Run the training script from the repository root:

```bash
python -m scripts.final_training
```

The current training configuration uses:

| Parameter | Value |
|---|---:|
| Learning parameter `tau` | -0.001 |
| Latent factor parameter `k` | 1000 |
| Regularization parameter `lambda_` | 0.01 |
| Parameter `alpha` | 0.1 |
| Batch size | 10,000 |
| Training epochs | 15 |

The training script saves the model checkpoint under the `output/` directory.

Expected checkpoint:

```text
output/model-0.001_1000.0_0.01_0.1_15.pickle
```

The exact output depends on the configuration and script version.

### Evaluate a trained model

To check the supported evaluation arguments:

```bash
python -m scripts.test_final_model --help
```

To evaluate the 15-epoch checkpoint using the dataset:

```bash
python -m scripts.test_final_model --model output/model-0.001_1000.0_0.01_0.1_15.pickle --dataset output/ctg-public-xml.sds
```

For full-dataset evaluation, inspect the available arguments before running:

```bash
python -m scripts.eval_full --help
```

**Important:** Evaluation results are comparable only when the dataset split, number of evaluated samples, and evaluation procedure are consistent.

## 8. Project Structure

```text
clinical-trial-keyword-recommendation/
├── ml/
│   └── mini_batch_relu_latent_factor_model.py
├── scripts/
│   ├── build_mesh_counts.py
│   ├── eval_full.py
│   ├── final_training.py
│   ├── search_alpha.py
│   ├── search_params.py
│   ├── split_data.py
│   ├── test_final_model.py
│   └── tune_small_vocab.py
├── output/
│   └── ctg-public-xml.sds
├── requirements.txt
└── README.md
```

The `output/` directory may also contain generated checkpoints and intermediate files. Large datasets and model checkpoints should generally not be committed to GitHub.

## 9. Memory Optimization

The implementation includes memory optimizations intended to make training and evaluation more practical on systems with limited RAM.

Key changes include:

- Sparse representation of the trial–keyword matrix instead of a dense matrix.
- `float32` data types where appropriate instead of `float64`.
- Sparse–dense matrix operations.
- Row-block processing of the neighbor-weight matrix.
- Running totals during MSE evaluation to avoid unnecessary memory allocation.

These changes target memory usage while preserving the underlying model formulation and gradient-update calculations.

## 10. Experimental Results

### Training

The full-vocabulary model was trained for 15 epochs.

| Metric | Result |
|---|---:|
| Initial training loss | 8.71816557 |
| Final training loss | 7.94570891 |

### Validation

The following results were obtained on the first 5,000 validation samples.

| Metric | Result |
|---|---:|
| MSE | 0.00102253 |
| MSER | 0.00153839 |
| APTK | 0.59007 |
| APRK | 0.001667 |

These results represent a partial validation evaluation, not a full validation-set or test-set evaluation.

An earlier 60-epoch baseline reported MSE = 0.001021, MSER = 0.001548, APTK = 0.5091, and APRK = 0.001257 on the full 61,018-sample test set. Since the evaluation splits and sample counts differ, these values should not be directly compared to claim an improvement.

## 11. Reproducibility Notes

- Run commands from the repository root.
- Use the expected Python environment and dependencies.
- Ensure that the dataset and preprocessing format are consistent.
- Keep training and evaluation splits consistent when comparing results.
- Record model parameters, epoch count, and evaluation sample count for each experiment.
- Do not interpret results from a partial validation sample as full test performance.

## 12. Troubleshooting

**Dataset not found**

Verify that the dataset exists at `output/ctg-public-xml.sds`.

**Module import errors**

Run commands from the repository root and ensure that the virtual environment is activated.

**Out-of-memory errors**

Check that the memory-optimized implementation is being used. Reduce the batch size only if the training script supports changing it, and record the change when comparing results.

**Checkpoint not found**

Verify that training completed successfully and that the checkpoint filename matches the selected training configuration.

## 13. References

- Original implementation: https://github.com/xiaomutt/xliniq
- Project repository: https://github.com/nehaha21/clinical-trial-keyword-recommendation
- ClinicalTrials.gov: https://clinicaltrials.gov/
