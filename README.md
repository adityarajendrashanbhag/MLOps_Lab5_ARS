# Lab 5: LLM Data Pipeline

## Project Overview

This project contains a modified version of Lab 5 focused on building an end-to-end LLM data pipeline for causal language modeling.

The notebook uses:

- `IMDB` movie reviews as the text dataset
- `DistilGPT-2` as the tokenizer
- `PyTorch DataLoader` for batch preparation

The goal is to show how raw text is transformed into fixed-length token sequences that can be used to train a GPT-style language model.

## What The Pipeline Does

The notebook in [`Lab1.ipynb`](./Lab1.ipynb) performs the following steps:

1. Loads a subset of the IMDB training split.
2. Initializes the `distilgpt2` tokenizer.
3. Tokenizes raw text reviews into token IDs.
4. Analyzes tokenized sequence lengths.
5. Compares multiple block sizes (`64`, `128`, `256`).
6. Concatenates tokenized text and chunks it into fixed-length blocks.
7. Builds a PyTorch `DataLoader`.
8. Verifies the final batch structure for causal language modeling.

## Why This Version Is Different

This lab was modified from the original reference version in the following ways:

- The dataset was changed from `WikiText-2` to `IMDB`.
- The tokenizer was changed from `gpt2` to `distilgpt2`.
- Additional analysis was added for token lengths.
- Block-size comparison was added to better explain sequence construction.

These changes make the notebook more clearly aligned with a practical LLM data pipeline while also satisfying the requirement that the submission should not be identical to the source repo.

## Project Structure

- [`Lab1.ipynb`](./Lab1.ipynb): Main notebook containing the full LLM data pipeline.

## Requirements

Install the following Python packages before running the notebook:

- `datasets`
- `transformers`
- `torch`

Example install command:

```bash
pip install datasets transformers torch
```

## Expected Output

After running the notebook, you should see:

- the number of IMDB reviews loaded
- a sample text preview
- tokenizer information
- token ID output from a sample review
- token length statistics
- estimated training sequence counts for different block sizes
- the number of final language-modeling sequences
- batch tensor shapes such as `torch.Size([8, 128])`

## Screenshots

Add screenshots here before submission.

### Screenshot 1: Dataset Loading

![Dataset Loading Placeholder](<img width="600" height="236" alt="image" src="https://github.com/user-attachments/assets/97bfa326-a487-4ce3-a8dc-dd291d62ec2d" />)

### Screenshot 2: Token Length Analysis

![Token Length Analysis Placeholder](<img width="477" height="123" alt="image" src="https://github.com/user-attachments/assets/0820e376-b97e-4dc6-83b1-35a562c42ac8" />)

### Screenshot 3: Final Batch Output

![Final Batch Output Placeholder](<img width="600" height="135" alt="image" src="https://github.com/user-attachments/assets/74b73479-40c7-4acc-8350-fbfb9ff29179" />)

## Notes

- The current notebook is intended as a data preprocessing pipeline, not a full model-training notebook.
- The final batches are prepared in a format suitable for decoder-only language models.
- If the screenshot files are not added, the image links will simply remain as placeholders.
