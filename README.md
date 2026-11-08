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

_To be added: loss curve, smoke tests, classification report, confusion matrix, and the comparison of baseline and fine-tuned accuracy._

## Credits

The notebook and dataset come from [The-Gen-Academy/5A-Fine-Tune-a-Support-Ticket-Router](https://github.com/The-Gen-Academy/5A-Fine-Tune-a-Support-Ticket-Router).
