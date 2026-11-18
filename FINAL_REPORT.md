# Pothole Detection: Experimental Report

**Task:** One-class pothole object detection from road images  
**Selected working model:** Semi-supervised YOLO11n student, 40 epochs  
**Working confidence threshold:** 0.368

## Executive summary

This project established a supervised YOLO11n baseline, audited and cleaned
the labeled data split, used a separate unlabeled image collection for a
pseudo-labeling experiment, and evaluated the resulting students on
validation. The 30- and 40-epoch semi-supervised students performed better
than the supervised 40-epoch baseline on the reported validation metrics.
Their results are close: the 30-epoch student has marginally higher F1, while
the 40-epoch student has higher recall and mAP. The 40-epoch model was chosen
as the working candidate, with confidence 0.368 selected as the operating
threshold.

A controlled common-corruption study also found that the model is sensitive to
strong image degradation, particularly additive Gaussian noise and blur. One
frozen moderate-noise condition was evaluated on the test split, but that
result is exploratory: the test split had already been evaluated for an
earlier supervised model.

These are useful project findings, not proof of real-world performance. The
study used one training seed, a single labeled dataset, synthetic corruptions,
and no untouched external test set. XAI and gradient-based adversarial testing
were not conducted.

## Dataset and split preparation

The labeled dataset contains one class, `pothole`. The source export notes
that images were stretched/resized to 640x640; their original aspect ratios
cannot be recovered from these supplied images. The source dataset was left
unchanged. A grouped copy was prepared for training and evaluation:

| Split | Images |
| --- | ---: |
| Train | 3,349 |
| Validation | 394 |
| Test | 197 |
| **Total** | **3,940** |

The original labeled data contained cross-split duplicate candidates. Related
images were grouped into a clean split; the resulting audit reported no
cross-split duplicate candidates. The audit and split files are in
[reports/](./reports/) and [data/leakage_clean_split/](./data/leakage_clean_split/).

The additional unlabeled collection contained 556 readable images. The audit
found no exact overlaps and 222 perceptual-hash near-match pairs, which were
treated as candidates requiring review, not as confirmed duplicates. Eleven
distinct validation-overlap candidates and three confirmed test matches were
excluded from pseudo-labeling. The resulting pseudo-label set used 434 of the
542 eligible images and contained 634 boxes. The 108 eligible images with no
detections at the chosen threshold were omitted rather than treated as
confirmed negative images. The
[pseudo-label summary](./data/pseudo_labels/yolo11n-40epoch-conf060/pseudo_label_summary.json)
records the settings and counts. Training-overlap candidates were not
automatically excluded, so some pseudo-label images may be redundant with
training data.

## Training and evaluation method

- Detector: Ultralytics YOLO11n transfer learning.
- Runtime: Python 3.12.10, PyTorch 2.11.0+cu128, Ultralytics 8.4.209.
- Hardware: NVIDIA GeForce RTX 3050 Laptop GPU, 4 GB VRAM.
- Image size: 640; supervised and semi-supervised runs used batch size 4 and
  seed 42. The semi-supervised students used two workers and `lr0=0.001`.
- Model selection and operating-threshold selection used validation data.
  The test split was not used for training.
- mAP is threshold-swept. Precision, recall, F1, and TP/FP/FN are operating
  point measures at the stated confidence threshold and IoU matching
  threshold 0.50. Detection confusion matrices include a background
  row/column; they do not have a meaningful image-classification true-negative
  count.

The supervised 40-epoch model was used as the teacher. On validation, teacher
confidence 0.60 had precision 0.956 and recall 0.648; it was selected to
favor high-precision pseudo-labels. Each student duration was trained
independently from that same teacher using the same fixed pseudo-label set.
No student was resumed from another student.

## Supervised and semi-supervised validation results

Each threshold below was selected from that run's validation F1 curve. Values
are rounded; each duration represents one training run.

| Model | Epochs | Confidence | Precision | Recall | F1 | mAP@0.50 | mAP@0.50:0.95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Supervised baseline | 20 | 0.442 | 0.854 | 0.754 | 0.801 | 0.862 | 0.535 |
| Supervised baseline | 30 | 0.414 | 0.868 | 0.810 | 0.838 | 0.897 | 0.573 |
| Supervised baseline | 40 | 0.374 | 0.855 | 0.826 | 0.840 | 0.909 | 0.594 |
| Semi-supervised student | 20 | 0.460 | 0.876 | 0.821 | 0.848 | 0.908 | 0.602 |
| Semi-supervised student | 30 | 0.384 | 0.890 | 0.857 | **0.873** | 0.924 | 0.630 |
| Semi-supervised student | 40 | 0.368 | 0.877 | **0.866** | 0.872 | **0.925** | **0.636** |

The 30- and 40-epoch students were the strongest validation runs. The 30-epoch
student's F1 exceeds the 40-epoch student's by about 0.001; the 40-epoch
student has higher recall and mAP. The improvement is promising for this
dataset, but a single run per duration does not establish that pseudo-labeling
will consistently improve performance.

### Operating-threshold tradeoff

The 40-epoch student's F1 curve peaks near confidence 0.368. Lower thresholds
increase recall while adding false positives:

| Confidence | Precision | Recall | F1 | TP | FP | FN |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.368 | 0.877 | 0.866 | 0.872 | 907 | 127 | 140 |
| 0.20 | 0.780 | 0.908 | 0.839 | 951 | 269 | 96 |
| 0.15 | 0.738 | 0.918 | 0.818 | 961 | 341 | 86 |

Confidence 0.368 is the user's selected **working** operating point because it
has the highest F1 and fewer false alarms. It is not the threshold that
minimizes missed potholes; 0.20 or 0.15 may be more appropriate if the
application later prioritizes recall enough to accept their larger false
alarm counts.

### Validation error review

At their respective F1 thresholds, the standalone FP16 qualitative review
counted 148 missed boxes and 104 false positives for the 30-epoch student,
versus 135 misses and 130 false positives for the 40-epoch student. Of the
missed boxes, 110 were common to both, 38 were unique to the 30-epoch review,
and 25 were unique to the 40-epoch review (box IoU >= 0.90). Most misses were
small: at most 5% of image area for 91% and 88% of misses, respectively.

These review counts differ slightly from the official evaluator because the
standalone prediction and matching paths differ. They are for qualitative
case review; use `evaluate.py` metrics for reported comparisons. The
[30-epoch review](./runs/detect/pothole-yolo11n-semi-r1-conf060-30ep-20261116-val-error-review/summary.json)
and
[40-epoch review](./runs/detect/pothole-yolo11n-semi-r1-conf060-40ep-20261116-val-error-review/summary.json)
contain per-image CSVs and annotated samples.

## Exploratory test evaluation

The selected 40-epoch student at confidence 0.368 was evaluated once on the
test split:

| Split/result | Precision | Recall | F1 | TP | FP | FN | mAP@0.50 | mAP@0.50:0.95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Test, clean | 0.888 | 0.837 | 0.862 | 334 | 42 | 65 | 0.920 | 0.667 |

This test split had already been evaluated for an earlier supervised
20-epoch model. Therefore this result is descriptive and exploratory, not an
untouched-holdout estimate. It was not used to revise the chosen checkpoint
or threshold. A new independent test set is needed for an unbiased final
performance claim.

## Inference-time common-corruption robustness

The robustness study applied each corruption independently to deterministic,
nested subsets of validation images at 10%, 25%, 50%, and 100% coverage (seed
42). The operating threshold remained fixed at 0.368. Corrupted images were
saved as lossless PNG copies; source images and labels were not modified.
These are synthetic common corruptions, not gradient-based adversarial
examples or training-data poisoning.

| Corruption | Mild | Moderate | Strong |
| --- | --- | --- | --- |
| Low light | Brightness factor 0.8 | 0.6 | 0.4 |
| Gaussian blur | Radius 0.5 px | 1.0 px | 2.0 px |
| JPEG compression | Quality 70 | 40 | 15 |
| Gaussian noise | Std. dev. 5 | 15 | 30 |

Noise standard deviations are in 8-bit pixel intensity units (0-255); the
equivalent normalized magnitudes are 5/255, 15/255, and 30/255.

The full validation matrix contains 49 rows: one clean condition and 48
corruption/coverage conditions. At full coverage, strong Gaussian noise
(standard deviation 30) had recall 0.368, 662 missed boxes, and mAP@0.50
0.537. Strong blur (radius 2 px) had recall 0.685, 330 missed boxes, and
mAP@0.50 0.844. Strong low light and JPEG quality 15 each reduced recall to
about 0.77.

At moderate Gaussian noise (standard deviation 15) and 25% validation
coverage, recall fell from 0.866 to 0.806, missed boxes rose from 140 to 203,
and mAP@0.50 fell from 0.925 to 0.882.

One frozen test condition used moderate Gaussian noise at 25% coverage (50 of
197 images):

| Test condition | Precision | Recall | F1 | TP | FP | FN | mAP@0.50 | mAP@0.50:0.95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Clean | 0.888 | 0.837 | 0.862 | 334 | 42 | 65 | 0.920 | 0.667 |
| Moderate noise, 25% | 0.893 | 0.774 | 0.830 | 309 | 37 | 90 | 0.860 | 0.592 |

This robustness result is also exploratory because the test split was
previously used for model evaluation.

The detailed
[validation robustness matrix](./runs/detect/pothole-yolo11n-semi-r1-conf060-40ep-20261116-robustness-valid-20261117/robustness_summary.csv)
and
[frozen test-condition results](./runs/detect/pothole-yolo11n-semi-r1-conf060-40ep-20261116-robustness-test-noise-moderate-25pct/robustness_summary.json)
preserve all measured values.

## Conclusions

1. A clean grouped split and validation-based model selection provide a more
   defensible comparison than the original split. Near-duplicate exclusions
   were also applied before pseudo-label training.
2. The pseudo-labeling experiment showed promising validation results. The
   30- and 40-epoch students improved over the supervised 40-epoch baseline
   on F1 and mAP in these runs, though repeat-seed experiments would be needed
   to determine whether the gains are stable.
3. The 40-epoch student is a reasonable working candidate when considering
   recall and mAP. The chosen confidence threshold of 0.368 balances precision
   and recall better than the lower thresholds, but it does not minimize
   missed potholes.
4. Strong blur and especially strong additive noise substantially degraded
   detection. The moderate-noise test result suggests a measurable effect at
   25% coverage, but it is exploratory and does not establish real-world
   robustness.

## Limitations and remaining work

- The test split is not untouched; final test results must be labeled
  exploratory. Use new, independently collected and annotated road images for
  stronger generalization claims.
- Training and corruption experiments used one fixed seed. Repeat training
  with multiple seeds before claiming stable performance improvements.
- Corruption strengths are controlled synthetic settings, not calibrated to a
  particular camera, road environment, or operational threat model.
- The unlabeled images were not manually verified box-by-box; pseudo-label
  omissions are not confirmed negatives, and training-overlap candidates may
  remain.
- XAI was deferred. No gradient-based adversarial attack or training-data
  poisoning experiment was conducted.
- Keep CC BY 4.0 attribution for the labeled Roboflow dataset in any
  presentation or redistribution.

## Reproducibility entry points

- [Project workflow and setup](./README.md)
- [Labeled-data audit](./dataset_audit.py) and [clean split preparation](./prepare_clean_split.py)
- [Pseudo-label generation](./prepare_pseudo_labels.py)
- [Semi-supervised student training](./train_semi_supervised.py)
- [Official evaluation](./evaluate.py)
- [Validation error review](./analyze_validation_errors.py)
- [Common-corruption robustness runner](./evaluate_corruption_robustness.py)
- Selected checkpoint:
  [40-epoch semi-supervised `best.pt`](./runs/detect/pothole-yolo11n-semi-r1-conf060-40ep-20261116/weights/best.pt)
