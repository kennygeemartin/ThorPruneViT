"""Reproducible illustrative score simulation; does not run a trained model."""
from pathlib import Path
import csv
import json
import math
import random
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'simulation'
LABELS = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
          'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
          'Emphysema', 'Fibrosis', 'Pleural Thickening', 'Hernia']
SEEDS = [42, 123, 456]
N = 2000
THRESHOLD = 0.5


def save_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metrics(y, scores):
    pred = [s >= THRESHOLD for s in scores]
    tp = sum(a == 1 and b for a, b in zip(y, pred))
    tn = sum(a == 0 and not b for a, b in zip(y, pred))
    fp, fn = len(y) - sum(y) - tn, sum(y) - tp
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn)
    # Rank-sum AUROC with average ranks for ties.
    ordered = sorted(zip(scores, y))
    rank_sum = 0.0
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][0] == ordered[i][0]:
            j += 1
        rank_sum += ((i + 1 + j) / 2) * sum(v for _, v in ordered[i:j])
        i = j
    pos = sum(y)
    auc = (rank_sum - pos * (pos + 1) / 2) / (pos * (len(y) - pos))
    assert tp + tn + fp + fn == len(y)
    result = dict(auroc=auc, accuracy=(tp + tn) / len(y), precision=precision,
                  sensitivity=recall, specificity=tn / (tn + fp),
                  f1=2 * tp / (2 * tp + fp + fn), tp=tp, tn=tn, fp=fp, fn=fn,
                  positive_count=pos, negative_count=len(y) - pos)
    assert all(0 <= result[k] <= 1 for k in ['auroc', 'accuracy', 'precision', 'sensitivity', 'specificity', 'f1'])
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, raw, aggregate = [], [], []
    for seed in SEEDS:
        rng = random.Random(seed)
        by_disease = []
        all_y, all_s = [], []
        exact = [True] * N
        for disease in LABELS:
            # Identical assumptions for every label: no disease-specific claims.
            y = [int(rng.random() < 0.15) for _ in range(N)]
            logits = [rng.gauss(1.0 if label else -1.0, 1.0) for label in y]
            scores = [1 / (1 + math.exp(-v)) for v in logits]
            m = metrics(y, scores)
            rows.append(dict(result_type='SIMULATED', seed=seed, disease=disease, **m))
            by_disease.append(m)
            all_y.extend(y); all_s.extend(scores)
            for i, (label, score) in enumerate(zip(y, scores)):
                exact[i] &= (score >= THRESHOLD) == bool(label)
                raw.append(dict(result_type='SIMULATED', seed=seed,
                                synthetic_sample_id=i, disease=disease,
                                synthetic_label=label, synthetic_score=score))
        micro = metrics(all_y, all_s)
        aggregate.append(dict(result_type='SIMULATED', seed=seed,
                              macro_auroc=statistics.mean(m['auroc'] for m in by_disease),
                              macro_f1=statistics.mean(m['f1'] for m in by_disease),
                              macro_precision=statistics.mean(m['precision'] for m in by_disease),
                              macro_sensitivity=statistics.mean(m['sensitivity'] for m in by_disease),
                              macro_specificity=statistics.mean(m['specificity'] for m in by_disease),
                              hamming_accuracy=micro['accuracy'], micro_f1=micro['f1'],
                              micro_auroc=micro['auroc'], exact_match_accuracy=sum(exact)/N))
    summary = []
    keys = ['auroc', 'accuracy', 'precision', 'sensitivity', 'specificity', 'f1']
    for disease in LABELS:
        runs = [r for r in rows if r['disease'] == disease]
        row = dict(result_type='SIMULATED', disease=disease, samples_per_seed=N)
        for k in keys:
            row[k + '_mean'] = statistics.mean(r[k] for r in runs)
            row[k + '_sd'] = statistics.stdev(r[k] for r in runs)
        summary.append(row)
    save_csv(OUT/'disease_wise_results.csv', summary)
    save_csv(OUT/'per_seed_results.csv', rows)
    save_csv(OUT/'aggregate_results.csv', aggregate)
    save_csv(OUT/'labels_and_scores.csv', raw)
    assumptions = dict(result_type='SIMULATED', trained_model_used=False,
                       description='Illustrative detection-score simulation, not model inference.',
                       seeds=SEEDS, samples_per_seed=N, threshold=THRESHOLD,
                       label_probability=0.15, positive_logit_mean=1.0,
                       negative_logit_mean=-1.0, logit_standard_deviation=1.0,
                       disease_specific_assumptions=False,
                       dependence='Labels and score noise independent across samples and diseases.',
                       uncertainty='SD measures simulation variability across seeds, not clinical uncertainty.')
    (OUT/'protocol.json').write_text(json.dumps(assumptions, indent=2), encoding='utf-8')
    lines = ['# Simulated disease-wise detection results', '',
             'Illustrative synthetic-score simulation. These values are generated from specified probability distributions; they are not predictions from ThorPruneViT or clinical evaluation.', '',
             'Three seeds (42, 123, 456), 2,000 synthetic samples per seed, 14 disease labels, and threshold 0.5. Each label has assumed probability 0.15. Positive and negative logits follow normal distributions with means +1 and -1 and standard deviation 1. The same assumptions apply to every disease. Differences between diseases reflect random sampling.', '',
             'Values below are mean ± sample SD across simulation seeds. AUROC is on the 0–1 scale; other values are percentages.', '',
             '| Disease | AUROC | Accuracy (%) | Precision (%) | Sensitivity (%) | Specificity (%) | F1 (%) |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        vals = [f"{r['auroc_mean']:.3f} ± {r['auroc_sd']:.3f}"]
        vals += [f"{100*r[k+'_mean']:.2f} ± {100*r[k+'_sd']:.2f}" for k in keys[1:]]
        lines.append('| ' + r['disease'] + ' | ' + ' | '.join(vals) + ' |')
    lines += ['', '## Aggregate simulation results', '', '| Metric | Mean ± SD |', '|---|---:|']
    for k in list(aggregate[0])[2:]:
        values = [r[k] for r in aggregate]
        lines.append(f'| {k} | {statistics.mean(values):.4f} ± {statistics.stdev(values):.4f} |')
    lines += ['', '## Results text for a simulation section', '',
              'A reproducible illustrative simulation generated synthetic labels and detection scores for 14 thoracic disease categories across three random seeds. Disease-wise AUROC, accuracy, precision, sensitivity, specificity, and F1 were computed from the generated labels and scores. The tables summarize variability across simulation runs under identical assumptions for all disease categories. This simulation demonstrates the reporting workflow; model performance is evaluated separately through inference on labelled images.', '',
              'Reproduce with `python scripts/generate_simulated_detection_results.py`. Raw labels and scores, confusion counts, per-seed metrics, aggregate metrics, and distribution assumptions accompany this report.']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
