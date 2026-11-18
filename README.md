# Pothole detection baseline

This project starts a supervised object-detection experiment for the one-class
`pothole` dataset. It uses Ultralytics YOLO11n transfer learning as a baseline. The labeled Roboflow export is licensed CC BY 4.0; retain
the dataset attribution in any report or redistributed results.

The completed experimental results and limitations are summarized in
[FINAL_REPORT.md](./FINAL_REPORT.md).

The source dataset is in the sibling folder
`..\resurse project 2\dataset1`. The audit reads it without modifying it.
Training and evaluation use a separate grouped copy at `data\leakage_clean_split`.

## 1. Prepare Python and CUDA

Install 64-bit Python 3.11 or 3.12 if it is not already installed. In PowerShell
from this project folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python train_baseline.py --check-cuda
```

The NVIDIA GeForce RTX 3050 with 4 GB VRAM is visible to the driver. This
baseline defaults to `yolo11n.pt`, 640-pixel images, and batch size 4 to limit
memory use. The first training run needs internet access to download pretrained
weights. If CUDA runs out of memory, retry with `--batch 2`; do not increase
image size until the baseline works.

## 2. Audit the labeled data before training

```powershell
python dataset_audit.py
```

The audit checks image/label pairing, YOLO box values, unreadable files, exact
duplicates, and perceptual near-duplicate candidates across and within all
splits. Perceptual matches (including 90-degree rotations) are candidates, not
proof that two images are duplicates. The CSV report is written to
`reports\source_dataset_audit.csv`.

The dataset export notes that it generated three augmented versions of source
images. That makes this audit especially important: related versions in
different splits can make evaluation results optimistic.

The audit found cross-split matches in the original dataset. The next command
groups every flagged exact/perceptual match together and creates a new,
approximately 85/10/5 split. It copies images and labels; it never edits the
downloaded source. Review the report and group manifest after creation, since a
perceptual-hash false positive can put unrelated images into the same group.

```powershell
python prepare_clean_split.py
python dataset_audit.py --dataset-root "data\leakage_clean_split" --report "reports\clean_split_audit.csv"
```

Confirm that the clean split audit reports **zero cross-split** duplicate
candidates before training. Within-split duplicates are acceptable; they do not
leak between training and evaluation. The generated `groups.csv` records each
image's original and new split.

## Preprocessing in this baseline

The dataset README says images were auto-oriented and resized to 640x640 using
stretching. The scripts keep these supplied images and YOLO-format boxes intact;
they do not apply another crop, resize, or normalization step. Ultralytics
handles image loading, resizing/letterboxing to `--imgsz`, and pixel
normalization during training and evaluation. Its standard training-time
augmentations are applied to training data only. Since the supplied images were
already stretched, their original aspect ratios cannot be recovered; later
datasets should preferably preserve aspect ratio and use consistent letterbox
preprocessing.

## 3. Train the supervised baseline

```powershell
python train_baseline.py
```

Training uses only the training images and validation split for model selection.
The held-out test split is not used by the training command. The script prints
the `best.pt` checkpoint path; preserve it for the final test evaluation.

Useful overrides:

```powershell
python train_baseline.py --epochs 10 --batch 2
python train_baseline.py --epochs 30 --batch 4 --name pothole-yolo11n-30-epoch-20261114
python train_baseline.py --dataset-root "D:\datasets\leakage_clean_split"
```

The 30- and 40-epoch runs used the same pretrained starting weights, seed,
image size, batch size, and augmentations as the 20-epoch run. Validation
mAP@0.50 and mAP@0.50:0.95 improved across these runs. The F1 peak also rose
slightly; each threshold below was selected separately from its own validation
F1 curve. These are validation comparisons, not test results.

| Run | Confidence | Precision | Recall | F1 | mAP@0.50 | mAP@0.50:0.95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 epochs | 0.442 | 0.854 | 0.754 | 0.801 | 0.862 | 0.535 |
| 30 epochs | 0.414 | 0.868 | 0.810 | 0.838 | 0.897 | 0.573 |
| 40 epochs | 0.374 | 0.855 | 0.826 | 0.840 | 0.909 | 0.594 |

## 4. Choose an operating threshold on validation data

Use validation data to inspect the precision/recall and F1 curves, and to choose
a confidence threshold appropriate for the application. This command runs
Ultralytics' threshold-swept mAP evaluation at its standard low confidence floor
and a second evaluation at the chosen threshold. The second run's confusion
matrix and precision/recall/F1 are all computed at that operating point
(IoU 0.50):

```powershell
python evaluate.py --weights "runs\detect\pothole-yolo11n-40-epoch-20261115\weights\best.pt" --split val --conf 0.374
```

The default `--conf 0.25` is only a starting point, not a threshold selected
for this dataset. For the current 40-epoch checkpoint the validation F1 curve
peaks near 0.374. Inspect `BoxF1_curve.png` in the `-map` output and choose
an operating threshold based on the F1 peak and whether missed potholes or
false alarms matter more. Each call writes separate `-map` and
`-conf-*` output folders under `runs\detect`; use the confusion matrix and
`metrics_summary.json` from the `-conf-*` folder. The `-map` folder contains
the precision/recall/F1 curves and threshold-swept mAP. Its confusion matrix
uses the low confidence floor needed for mAP and is not the selected
operating-point matrix.

The detection confusion matrix has a background row/column and depends on
confidence and IoU matching; it is not an ordinary image-classification matrix
with meaningful true negatives. Report its threshold with its precision,
recall, and F1. The mAP scores summarize performance across confidence
thresholds, so do not compare them directly with the single-threshold matrix.

## 5. Review validation errors before changing the training recipe

Review the validation false positives and missed boxes before changing the
training recipe:

```powershell
python analyze_validation_errors.py --weights "runs\detect\pothole-yolo11n-40-epoch-20261115\weights\best.pt" --conf 0.374
```

This writes a per-image CSV, summary, and annotated examples under
`runs\detect\pothole-yolo11n-40-epoch-20261115-val-error-review`. Red boxes are
ground truth, green boxes are matched detections, and yellow boxes are false
positives. The report's standalone predictions support qualitative review;
use `evaluate.py` for official metrics because CPU/GPU numerical differences
can change detections near the confidence threshold.
If GPU memory is occupied or out of memory, rerun the review with
`--device cpu --batch 1`; use the official evaluator's metrics rather than the
standalone review counts for model comparisons. The review uses FP16 inference
on CUDA to match Ultralytics validation and reduce memory use on the 4-GB GPU;
the CPU fallback uses FP32.

The latest CUDA review found 179 missed boxes; 97 (54%) covered at most 1% of
an image and 158 (88%) at most 5%. These are object-area fractions from the
qualitative review pass, not official detection metrics. Manual spot-checks of
three representative images judged the labels mostly sound, with the remaining
small misses less important for this project.

The initial CUDA error-review attempt used Ultralytics prediction's default
full precision and ran out of memory. A single-image CUDA smoke test succeeded,
and the full validation review completed after setting FP16 explicitly. This
was a memory-precision mismatch in the standalone review, not a missing CUDA
device or CPU-only PyTorch installation.

## 6. Completed longer supervised comparison

The 40-epoch run changed only training duration relative to the 30-epoch run.
This is the command used to create it:

```powershell
python train_baseline.py --epochs 40 --batch 4 --name pothole-yolo11n-40-epoch-20261115
```

The 40-epoch checkpoint is the current supervised candidate based on validation
mAP and F1. No test-set result was used to make this selection.

## 7. Final held-out evaluation

After selecting the model and confidence threshold using validation, evaluate
the test split once with that threshold. The test set was already evaluated
for the 20-epoch checkpoint earlier in this project, so it is no longer an
untouched holdout for model selection. Do not use it to choose between
checkpoints; use a new independent test set if an unbiased final estimate is
required.

```powershell
python evaluate.py --weights "runs\detect\pothole-yolo11n-40-epoch-20261115\weights\best.pt" --split test --conf 0.374
```

Use validation results for model or threshold decisions. Later semi-supervised
runs and adversarial robustness experiments should use the same fixed test set
and report clean vs. attacked metrics separately.

The selected semi-supervised 40-epoch checkpoint was also evaluated once on
the test split at confidence 0.368, after model and threshold selection. It
reported precision 0.888, recall 0.837, F1 0.862, mAP@0.50 0.920,
mAP@0.50:0.95 0.667, and 65 false negatives / 42 false positives across 197
images. Treat this as an **exploratory** result: the test split had already
been evaluated for an earlier supervised checkpoint, so it is not an
untouched holdout. Do not use it to revise the selected checkpoint or
threshold; a new independent test set is needed for an unbiased final claim.

## 8. First semi-supervised experiment (pseudo-labeling)

This first round uses self-training: the supervised 40-epoch checkpoint acts as
a teacher to label the separate unlabeled image folder, then a student is
fine-tuned on the original labeled training split plus those pseudo-labels.
The validation and test splits remain evaluation-only.

The unlabeled overlap audit found 11 distinct validation candidates and three
confirmed test matches. All 14 are excluded from pseudo-label generation to
avoid validation/test leakage. This exclusion is conservative: perceptual-hash
matches are candidates and may include false positives. Images are not moved or
modified. Matches against the labeled training split are not automatically
excluded; they can be redundant examples and should be considered when
interpreting the experiment.

Before generating labels, validation measurements for the supervised teacher
showed the confidence tradeoff below. Confidence 0.60 was selected to favor
high-precision pseudo-labels while retaining more recall than the 0.70 setting:

| Confidence | Validation precision | Validation recall |
| ---: | ---: | ---: |
| 0.55 | 0.937 | 0.709 |
| 0.60 | 0.956 | 0.648 |
| 0.70 | 0.979 | 0.499 |

Generate the pseudo-label dataset:

```powershell
python prepare_pseudo_labels.py --conf 0.60
```

By default this uses the 40-epoch teacher, 640-pixel images, CUDA device 0, and
the audited sibling unlabeled folder. It writes a separate dataset under
`data\pseudo_labels\yolo11n-40epoch-conf060`, with a manifest, summary, and
review contact sheets. The 542 eligible images produced 434 pseudo-labeled
images containing 634 boxes; 108 images with no detections at this threshold
are omitted, not treated as confirmed negatives. The review sheets prioritize
the lowest-confidence accepted predictions and are a sample for inspection,
not a complete review of every pseudo-box.

Fine-tune a student from the supervised teacher:

```powershell
python train_semi_supervised.py --epochs 20 --batch 4 --workers 2 --device 0 --lr0 0.001 --name pothole-yolo11n-semi-r1-conf060-20261116
python train_semi_supervised.py --epochs 30 --batch 4 --workers 2 --device 0 --lr0 0.001 --name pothole-yolo11n-semi-r1-conf060-30ep-20261116
python train_semi_supervised.py --epochs 40 --batch 4 --workers 2 --device 0 --lr0 0.001 --name pothole-yolo11n-semi-r1-conf060-40ep-20261116
```

The script validates the pseudo-label manifest, checks that validation/test
candidates are absent, combines the two training-image directories, and uses
only the existing validation split for checkpoint selection. It does not
evaluate the test split. To try another confidence threshold, give
`prepare_pseudo_labels.py` a new `--output-root`, then pass that folder as
`--pseudo-root` to the training script. Each duration above is an independent
run from the same supervised teacher, using the same fixed pseudo-label set
and training settings; do not resume one student from another.

Each model's operating threshold was selected from its own validation F1
curve; mAP is threshold-swept. The 30- and 40-epoch runs improve the overall
validation comparison. The 30-epoch run has the highest F1, while the 40-epoch
run has the highest recall and mAP. Choose between them based on the relative
cost of missed potholes versus false alarms, and review their validation errors
before settling on a deployment candidate.

| Run | Confidence | Precision | Recall | F1 | mAP@0.50 | mAP@0.50:0.95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Supervised, 40 epochs | 0.374 | 0.855 | 0.826 | 0.840 | 0.909 | 0.594 |
| Semi-supervised, 20 epochs | 0.460 | 0.876 | 0.821 | 0.848 | 0.908 | 0.602 |
| Semi-supervised, 30 epochs | 0.384 | 0.890 | 0.857 | **0.873** | 0.924 | 0.630 |
| Semi-supervised, 40 epochs | 0.368 | 0.877 | **0.866** | 0.872 | **0.925** | **0.636** |

These are validation results from one run, not evidence that semi-supervised
training consistently improves performance. The test split has not been used
for this experiment. The F1 difference between the 30- and 40-epoch runs is
small; the 40-epoch run's recall and mAP are higher, while the 30-epoch run's
precision and F1 are slightly higher. Keep the supervised checkpoint as the
control, and do not use the test split to select between these students.

The 40-epoch model-specific validation F1 curve peaked at confidence 0.368.
Its operating-point evaluation at that threshold reported precision 0.877,
recall 0.866, and F1 0.872; threshold-swept mAP was 0.925 / 0.636. Reproduce
that evaluation with:

```powershell
python evaluate.py --weights "runs\detect\pothole-yolo11n-semi-r1-conf060-40ep-20261116\weights\best.pt" --split val --conf 0.368368
```

Because missed potholes are considered more costly than false alarms, lower
confidence thresholds were also evaluated on validation. Lower thresholds
increase recall, but add false positives. The metrics below are from
`evaluate.py`; mAP is threshold-swept and therefore unchanged across these
operating points.

| Confidence | Precision | Recall | F1 | TP | FP | FN |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.368 (F1 peak) | 0.877 | 0.866 | 0.872 | 907 | 127 | 140 |
| 0.20 | 0.780 | 0.908 | 0.839 | 951 | 269 | 96 |
| 0.15 | 0.738 | **0.918** | 0.818 | 961 | 341 | **86** |

Confidence 0.20 finds 44 more boxes than the F1-peak operating point, with 142
additional false positives. Confidence 0.15 finds 10 more boxes than 0.20,
with 72 additional false positives. The current working operating point for
subsequent application work is confidence 0.368, selected for its higher F1
and lower false-alarm count. The lower thresholds remain recall-oriented
alternatives if the cost of missed potholes is later judged to outweigh the
extra false alarms. This is a validation-based working choice, not a claim of
final deployment suitability; do not tune the threshold on the test split.

Review the two students' per-image misses and false alarms at their own
validation F1 thresholds:

```powershell
python analyze_validation_errors.py --weights "runs\detect\pothole-yolo11n-semi-r1-conf060-30ep-20261116\weights\best.pt" --conf 0.384 --output "runs\detect\pothole-yolo11n-semi-r1-conf060-30ep-20261116-val-error-review"
python analyze_validation_errors.py --weights "runs\detect\pothole-yolo11n-semi-r1-conf060-40ep-20261116\weights\best.pt" --conf 0.368368 --output "runs\detect\pothole-yolo11n-semi-r1-conf060-40ep-20261116-val-error-review"
```

In the standalone FP16 review pass, the 30-epoch student had 148 missed boxes
and 104 false positives; the 40-epoch student had 135 missed boxes and 130
false positives. Matching the two miss lists by box IoU >= 0.90 found 110
misses common to both, 38 only for the 30-epoch model, and 25 only for the
40-epoch model. Most misses were small: 91% (30 epochs) and 88% (40 epochs)
covered at most 5% of their image area. Spot-checks show differences by image,
not a universal winner. The 40-epoch model trades more false alarms for fewer
misses overall. These standalone counts can differ slightly from `evaluate.py`
because of inference and matching details; use `evaluate.py` for reported
metrics and the annotated samples for qualitative review.

## 9. Inference-time common-corruption robustness

The robustness runner evaluates the selected 40-epoch student at the chosen
confidence threshold, defaulting to the validation split. It measures each
corruption independently across deterministic, nested subsets of 10%, 25%,
50%, and 100% of validation images. Unaffected images remain clean, and source
images and labels are never modified; transformed copies are saved losslessly
as PNG to avoid adding JPEG artifacts to the other corruption families. The
runner reports mAP@0.50, mAP@0.50:0.95, and precision/recall/F1 plus TP/FP/FN
at confidence 0.368.

The mild/moderate/strong parameters are brightness factors 0.8/0.6/0.4,
Gaussian-blur radii 0.5/1.0/2.0 pixels, JPEG qualities 70/40/15, and Gaussian
noise standard deviations 5/15/30 on the 0-255 pixel scale. Run the full
validation matrix with:

```powershell
python evaluate_corruption_robustness.py --conf 0.368368 --seed 42
```

The run writes `robustness_summary.csv` and `robustness_summary.json` under a
timestamped folder in `runs\detect`; each condition's evaluator outputs are
stored alongside them. After selecting one corruption, severity, and coverage
using validation, the runner permits exactly one frozen test condition, for
example:

```powershell
python evaluate_corruption_robustness.py --split test --families blur --severities moderate --coverages 0.25 --conf 0.368368
```

The completed validation matrix contains 49 rows: one clean baseline and 48
corruption/coverage conditions. The largest full-coverage drops were from
strong Gaussian noise (standard deviation 30; recall 0.368, 662 missed boxes,
mAP@0.50 0.537) and strong blur (radius 2 pixels; recall 0.685, 330 missed
boxes, mAP@0.50 0.844). Strong low light (brightness factor 0.4) and JPEG
quality 15 each reduced recall to about 0.77 at full coverage. At moderate
Gaussian noise (standard deviation 15) and 25% coverage, validation recall
fell from 0.866 to 0.806, misses rose from 140 to 203, and mAP@0.50 fell from
0.925 to 0.882.

Based on that validation result, one frozen test condition was evaluated:
moderate Gaussian noise on 25% of the test images (50 of 197). The clean test
baseline had recall 0.837, 65 missed boxes, mAP@0.50 0.920, and
mAP@0.50:0.95 0.667. Under corruption, recall was 0.774, misses were 90,
mAP@0.50 was 0.860, and mAP@0.50:0.95 was 0.592. Precision was 0.893
(clean: 0.888); false positives were 37 (clean: 42). This single robustness
result is exploratory because the test split had been evaluated previously.
The CSV/JSON summaries preserve all measured conditions.

The test split was previously used for another checkpoint, so test robustness
results are exploratory rather than an untouched-holdout estimate. These
common corruptions test input-quality robustness; they are distinct from
gradient-based adversarial attacks and training-data poisoning.

## Experiment guidance

- The unlabeled images are not used in these supervised runs. They are a
  candidate for further semi-supervised experiments; the first pseudo-labeling
  run and its results are documented above.
- The common-corruption robustness experiment above is not a gradient-based
  adversarial-attack or training-data-poisoning experiment. Treat either as a
  separate follow-up with a specified threat model and strength.
- Detection-aware XAI is a reasonable next experiment. Grad-CAM or EigenCAM
  can help visualize evidence for selected predicted boxes; validate
  explanations with perturbation/sanity checks rather than treating a heatmap
  alone as proof.
- Keep future adversarial evaluation and XAI implementations in their own
  scripts/modules, separate from this supervised training pipeline.
- Keep a run log with the checkpoint, settings, split audit, package versions,
  random seed, and metrics so comparisons remain reproducible.
