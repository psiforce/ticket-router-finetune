# Support Ticket Router: Fine-Tuning Qwen3-1.7B with LoRA

This is the Week 5 project for The Gen Academy. It turns `Qwen/Qwen3-1.7B-Base` into a small IT support-ticket router. The model reads a ticket and predicts exactly one queue label. It never writes a reply. Training is LoRA fine-tuning with [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory) on a free Colab T4 GPU.

## Routing labels

| Label | Queue |
|---|---|
| Active Directory | Identity and access: accounts, deactivation, permissions |
| Computer-Services | Endpoints and hardware: devices, printers, workstations |
| EOL | Lifecycle: end-of-life, retirement, decommissioning |
| Fileservice | File storage: shared drives, folders, file permissions |
| O365 | Microsoft 365: Outlook, Teams, mailbox, SharePoint |
| Software | Applications: installs, licensing, updates |
| Support general | General triage, for tickets that need human review |

## Repo contents

| File | What it is |
|---|---|
| [GUIDE.md](GUIDE.md) | Step-by-step guide to running the notebook in Colab, with key concepts and troubleshooting |
| `Finetune_Support_Ticket_Classifier_Qwen3.ipynb` | The course notebook |
| `support_tickets.csv` | Labelled tickets (`category_truth`, `text`). The notebook splits them 80/20 into train and validation sets. |

## Pipeline

```
CSV → stratified 80/20 split → ShareGPT TRAIN.json + dataset_info.json
    → LLaMA Board (LoRA SFT, fp16, T4) → adapter
    → merge_and_unload() → classify() smoke test
    → classification report + confusion matrix + letter-choice baseline comparison
```

## Results

There's a video walkthrough of about 5 minutes: [Week5_TicketRouter_Video.mp4](Week5_TicketRouter_Video.mp4). The screenshots are in the [submission doc](FinTune_Support-Ticket-Router.pdf).

| Metric | Value |
|---|---|
| Data | 585 tickets → 468 train / 117 held-out validation (stratified) |
| Training loss | 6.22 → 0.18 over about 90 steps (converged) |
| Smoke test | 4 of 5 correct. A new-user account request went to Support general |
| Baseline accuracy (untuned, letter-choice prompt) | 25% |
| **Fine-tuned accuracy** | **73.5%** (+49 pts), macro F1 0.70 |

The per-class results below come from the confusion matrix:

| Class | Correct | Recall |
|---|---|---|
| EOL | 12 of 12 | 100% |
| Fileservice | 23 of 28 | 82% |
| Support general | 24 of 30 | 80% |
| Computer-Services | 6 of 8 | 75% |
| O365 | 13 of 18 | 72% |
| Software | 6 of 12 | 50% |
| **Active Directory** | **2 of 9** | **22%** |

Active Directory is the weakest class. Five of its nine tickets were routed to Support general.

> **Note:** the notebook's `classification_report` and comparison chart print class names in the wrong order. scikit-learn sorts labels alphabetically, but `target_names` uses the notebook's own order. The numbers in the tables above were cross-checked against the confusion matrix, which is labelled correctly.

**Next steps:**
- Add more Active Directory examples or use class weighting.
- Train for more epochs.
- Add a confidence threshold that sends low-confidence tickets to human review.

## Credits

The notebook and dataset come from [The-Gen-Academy/5A-Fine-Tune-a-Support-Ticket-Router](https://github.com/The-Gen-Academy/5A-Fine-Tune-a-Support-Ticket-Router).
