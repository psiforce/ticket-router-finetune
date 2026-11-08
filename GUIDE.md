# Week 5: Fine-Tune a Support Ticket Router (Colab Guide)

This guide walks through the course notebook `Finetune_Support_Ticket_Classifier_Qwen3.ipynb` in Google Colab, one step at a time. It covers 28 cells.

Colab doesn't number cells, so each step below names the **section heading** you'll see in the notebook and the **first code line** of the cell to run. To jump between sections, open **View → Table of contents**.

The whole run takes about 2–3 hours. Most of that is waiting: training takes 30–60 minutes, and the baseline plus evaluation take 10–30 minutes.

---

## Before you start

1. Get `Finetune_Support_Ticket_Classifier_Qwen3.ipynb` and `support_tickets.csv`. Both are in this repo and in [the course repo](https://github.com/The-Gen-Academy/5A-Fine-Tune-a-Support-Ticket-Router).
2. Go to [colab.research.google.com](https://colab.research.google.com), then **File → Upload notebook**, and pick the `.ipynb`.
3. Do this **first**: **Runtime → Change runtime type → T4 GPU → Save**. If you change the runtime type later, it wipes everything you've installed or uploaded.

### How to add a new cell (for the checks in this guide)

Use any of these:

- Click **+ Code** in the toolbar. The cell is inserted below the cell you last clicked.
- Hover over the bottom edge of a cell, then click **+ Code**.
- Click the left margin of a cell, then press **Ctrl+M B** to add a cell below or **Ctrl+M A** to add one above.

Run a cell with **Shift+Enter** or the ▶ button. A `!` at the start of a line runs it as a shell command instead of Python.

---

## Step-by-step

| Section heading in Colab | First code line | Run it? |
|---|---|---|
| Install Dependencies | `%cd /content/` … `git clone` … `pip install -e .[torch,bitsandbytes]` | ✅ |
| Check GPU environment | `import torch` … `torch.cuda.is_available()` | ✅ |
| Update Identity Dataset | `NAME = "Llama-3"` … | Optional. Skip it |
| Prepare Support Ticket Dataset (INPUT REQUIRED) | long cell with `LLAMA_DATA_DIR` and `files.upload()` | ✅ Upload the CSV |
| Fine-tune model via LLaMA Board (INPUT REQUIRED) | `!GRADIO_SHARE=1 llamafactory-cli webui` | ✅ Train, then **stop** the cell |
| Review Training — Loss Curve (INPUT REQUIRED) | `ADAPTER_DIR = "...train_XXXX..."  # <-- CHANGE THIS` | ✅ Edit `ADAPTER_DIR` first |
| Merge Adapter Weights, Measure Baseline & Prepare for Inference | `CHOICES = "ABCDEFG"` … | ✅ |
| Define classify() and Smoke Test the Merged Model | `def classify(ticket_text, ...)` | ✅ |
| Evaluate the Fine-tuned Model on the Validation Split | loop that builds `y_pred`, then `classification_report(...)` | ✅ |
| Confusion matrix | seaborn heatmap | ✅ |
| Baseline vs Fine-tuned Comparison | bar-chart code | ✅ |

### Phase 1: Install Dependencies

This cell clones LLaMA-Factory into `/content/LLaMA-Factory` and runs `pip install -e .[torch,bitsandbytes]`.

- **It's done when** the cell shows a green ✓ and the output ends with `Successfully installed ...`. If you see a warning that says `pip's dependency resolver ...`, you can ignore it.
- If Colab shows **Restart session**, click it. Don't run the install cell again afterwards.
- To check it worked, run `!ls /content/LLaMA-Factory/data | head`. You should see `dataset_info.json` in the list.

### Check GPU environment

This cell checks `torch.cuda.is_available()`. It should pass without printing the setup-help message. If it fails, set the runtime to T4 GPU and redo Phase 1.

### Update Identity Dataset (optional)

This cell is leftover LLaMA-Factory boilerplate. It only edits `data/identity.json`, a "Llama-3, made by LLaMA Factory" chatbot persona, and the router doesn't use that file. Skipping it is fine, and so is running it. Just **never select `identity` in LLaMA Board**.

### Phase 2: Prepare Support Ticket Dataset

1. Run the cell, then click **Choose Files** under it and pick `support_tickets.csv`. The notebook text says "support_ticket.csv", but the code accepts any filename. If the button never appears, see [Troubleshooting](#troubleshooting).
2. The cell then does the following:
   - It reads the `category_truth` column (renamed to `label`) and the `text` column, and keeps only the 7 labels.
   - It makes a **stratified 80/20 split** with `random_state=42`.
   - It turns each training row into a 3-message conversation:
     - system: *"You are an IT helpdesk ticket routing assistant… respond with exactly one of the following categories: …"*
     - user: `Support Ticket: <text>`
     - assistant: `<label>`
   - It writes `/content/LLaMA-Factory/data/TRAIN.json` and **registers `support_tickets`** in `dataset_info.json`.
   - It saves the held-out rows to `/content/val_split.csv`.
3. **It's done when** the output shows `Written N records → .../TRAIN.json` plus the split counts. To check, run `!ls /content/LLaMA-Factory/data/TRAIN.json /content/val_split.csv`.

### Phase 3: Fine-tune via LLaMA Board

1. Run the cell and open the `https://xxxx.gradio.live` link it prints.
2. In the **Train** tab, set the following:
   - **Model name/path**: `Qwen/Qwen3-1.7B-Base`
   - **Finetuning method**: `lora`
   - **Stage**: Supervised Fine-Tuning
   - **Dataset dir**: `data`
   - **Dataset**: `support_tickets` only
   - **Compute type**: **fp16**. A T4 can't run bf16, and fp32 runs out of memory.
   - Leave everything else at its default.
3. Write down the **Output Dir**, for example `train_2026-11-08-14-30-00`.
4. Click **Start**. Clicking "Preview command" alone does **not** train. Within a few minutes you should see a progress bar and a loss chart. Ignore occasional "syntax error" pop-ups.
5. Keep the Colab tab open. Free Colab disconnects idle sessions and wipes `/content`.
6. When the UI shows **Training completed**, go back to Colab and **stop the LLaMA Board cell (■)**. Later cells can't run while that cell is still running.

### Phase 4: Review Training (Loss Curve)

1. Find the real adapter folder:
   ```
   !find /content/LLaMA-Factory/saves -name "trainer_log.jsonl"
   ```
2. Copy the printed path **without** `/trainer_log.jsonl` into `ADAPTER_DIR`, for example:
   ```python
   ADAPTER_DIR = "/content/LLaMA-Factory/saves/Qwen3-1.7B-Base/lora/train_2026-11-08-14-30-00"
   ```
3. Run the cell. It plots the loss and saves `/content/training_curve.png`. A final loss below 0.5 means training converged, and below 1.2 is moderate. A healthy curve drops and then levels off. **Take a screenshot.**

### Phase 5: Merge, Baseline, and Smoke Test

- **Merge Adapter Weights, Measure Baseline & Prepare for Inference**
  - **Baseline**: the untouched base model answers with a single letter A–G (A = Support general … G = Computer-Services). The cell scores those letters and prints a classification report on `val_split.csv`.
  - **Merge**: the cell attaches your LoRA adapter, calls `merge_and_unload()`, and saves the result to `/content/qwen3_merged`.
- **Define classify() and Smoke Test**: `classify()` uses the **same system prompt as training**, with greedy decoding and up to 10 tokens. It runs 5 obvious tickets. **All 5 should show OK. Take a screenshot.**
- If smoke tests fail, or everything comes out as "Support general", check three things: is `ADAPTER_DIR` correct, does the loss curve look healthy, and did you change the Template setting in LLaMA Board?

### Phase 6: Evaluate and Compare

1. Run the **evaluation loop**. It runs `classify` on every validation ticket, which takes several minutes.
2. Read the **classification report**, which shows precision, recall, F1, and support for each class. **Take a screenshot.**
3. Look at the **confusion matrix**. It saves `/content/confusion_matrix.png` and warns if Active Directory recall is below 0.85. **Take a screenshot.**
4. Look at **Baseline vs Fine-tuned**. It shows per-class F1, overall accuracy, and the delta between the two models, and saves `/content/baseline_vs_finetuned.png`. **Take a screenshot.**

---

## Key concepts

- **Base model and system prompt.** Qwen3-1.7B-*Base* was never instruction-tuned. Training it on (system prompt, ticket, label) examples teaches it to reply with exactly one label. Inference must use the **same prompt** it was trained on.
- **LoRA.** The 1.7B base weights stay frozen. Training only fits a small low-rank adapter (W + B·A). The adapter is a few MB and is saved in `ADAPTER_DIR`.
- **Merging.** Merging folds B·A back into the base weights. You end up with one standalone model and no adapter overhead at inference time.
- **Stratified split.** Each class keeps the same share in the training and validation sets. The 20% validation set is never trained on, so its score is an honest measure.
- **Letter-choice baseline.** An untrained model can't reliably spell the labels. Scoring the letters A–G tests what it *knows*, not how well it formats an answer, so the comparison with the fine-tuned model is fair.
- **Metrics.**
  - **Recall** is the share of each class's tickets that were routed correctly. It matters most for Active Directory, because a missed ticket means someone stays locked out.
  - **Precision** is the share of tickets routed to a queue that actually belong there.
  - In the **confusion matrix**, rows are true classes and columns are predictions. Off-diagonal cells show which queues get confused with each other.
- **Tuning.** Do this only if the first run is weak, and change one thing at a time:

  | What you see | What to change |
  |---|---|
  | Loss is still falling at the end | More epochs |
  | Loss jumps around | Lower learning rate |
  | Out of memory | Smaller batch size or cutoff length |
  | Model is underfitting | Higher LoRA rank |

---

## Troubleshooting

### The dataset cell spins and never shows "Choose Files"

1. Make sure every cell above it shows a green ✓. If **Install** is still running, the dataset cell is just waiting in the queue.
2. Scroll to the bottom of the cell's output, where the button appears.
3. If the cell is still stuck after about a minute, the browser is blocking the upload widget. That usually happens with third-party cookies blocked, or in Brave, Safari, or incognito mode.
   1. Stop the cell (■).
   2. Upload `support_tickets.csv` through the **📁 Files** panel into `/content`. Put it at the top level, not inside `LLaMA-Factory`.
   3. In the dataset cell, replace this line:
      ```python
      uploaded = files.upload()
      ```
      with this one:
      ```python
      uploaded = {"support_tickets.csv": open("/content/support_tickets.csv", "rb").read()}
      ```
   4. Run the cell again.

### `FileNotFoundError: ... /content/LLaMA-Factory/data/TRAIN.json`

LLaMA-Factory isn't installed in this session. Either the install never finished, or the session reset and wiped `/content`. Check with `!ls /content`. If there's no `LLaMA-Factory` folder, run the cells again from **Install Dependencies**, in order.

### `trainer_log.jsonl not found in '.../saves/...'`

1. Run `!find /content/LLaMA-Factory/saves -name "trainer_log.jsonl"`.
2. If it prints a path, copy that folder path into `ADAPTER_DIR`.
3. If it prints nothing, or says `No such file or directory`, training never ran or the session was reset. Rerun from Install if needed, then retrain:
   - Set **fp16**.
   - Click **Start**.
   - Wait for **Training completed**.

### Out of memory in the merge or evaluation cells

Use **Runtime → Restart session**. Installed packages are kept. Then rerun the Loss Curve cell, the Merge cell, and the cells after them.

---

## Submission checklist

Put these in a Google Doc:

- Screenshots of:
  - the loss curve
  - the 5/5 smoke tests
  - the classification report
  - the confusion matrix
  - the baseline vs. fine-tuned chart, with the delta
- 2–3 sentences interpreting the results

Submit it at https://forms.gle/cmB8mLtpjNCNsjrZ6. The deadline is **Sun Nov 8, 2026, 11:59pm PT**.

**The run counts as successful when** the loss curve levels off, all 5 smoke tests pass, and fine-tuned accuracy is well above the baseline.
