"""Grid eval: simulate cadences & delays from rolling edges, compare vs offline."""
from __future__ import annotations

import json
import pandas as pd
import numpy as np
from pathlib import Path
from football_ai.pathcrf_slot_mapping import person_slot_to_canonical_id, referee_slot_to_canonical_id

CADENCES = [10, 25, 50, 75, 100]
DELAYS = [75, 100, 125, 150]
FS, FE, TOL = 40, 724, 5
EMIT_FRAMES = 10
MIN_WARMUP = 50

def canon(s):
    if s is None: return None
    p = person_slot_to_canonical_id(str(s)); r = referee_slot_to_canonical_id(str(s))
    return str(p) if p else (str(r) if r else None)

def load_pairs(df, key='canonical'):
    pairs = {}
    for _, row in df.iterrows():
        fid = int(row['frame_id'])
        if key == 'canonical':
            cs, cd = str(row.get('canonical_src','')), str(row.get('canonical_dst',''))
            if cs == 'None' or cd == 'None': continue
            pairs[fid] = (cs, cd)
        else:
            cs, cd = canon(row.get('edge_src')), canon(row.get('edge_dst'))
            if cs and cd: pairs[fid] = (cs, cd)
    return pairs

def match(pred, ref):
    used = set(); n = 0
    for pf, (ps, pd) in sorted(pred.items()):
        for rf, (rs, rd) in ref.items():
            if rf in used: continue
            if ps == rs and pd == rd and abs(pf - rf) <= TOL:
                used.add(rf); n += 1; break
    p = n / len(pred) if pred else 0; r = n / len(ref) if ref else 0
    return n, p, r, 2*p*r/(p+r) if (p+r)>0 else 0

def simulate(rolling_df, cadence, delay):
    emitted = {}
    checkpoints = sorted(rolling_df['checkpoint_frame'].unique())
    selected = []
    for cp in checkpoints:
        if not selected or (cp >= MIN_WARMUP and cp - selected[-1] >= cadence):
            selected.append(cp)
    
    for cp in selected:
        cp = int(cp)
        emit_end = cp - delay
        emit_start = emit_end - EMIT_FRAMES + 1
        if emit_start > emit_end: emit_start = emit_end
        cp_rows = rolling_df[rolling_df['checkpoint_frame'] == cp]
        for _, row in cp_rows.iterrows():
            fid = int(row['frame_id'])
            if not (emit_start <= fid <= emit_end): continue
            cs, cd = str(row.get('canonical_src','')), str(row.get('canonical_dst',''))
            if cs == 'None' or cd == 'None': continue
            if fid not in emitted:
                emitted[fid] = (cs, cd)
    return emitted, selected

def dedup_edges(rolling_df, cadence):
    checkpoints = sorted(rolling_df['checkpoint_frame'].unique())
    selected = []
    for cp in checkpoints:
        if not selected or (cp >= MIN_WARMUP and cp - selected[-1] >= cadence):
            selected.append(cp)
    # Take only rows from selected checkpoints
    filtered = rolling_df[rolling_df['checkpoint_frame'].isin(selected)]
    dedup = filtered.sort_values('checkpoint_frame').drop_duplicates('frame_id', keep='last')
    pairs = {}
    for _, row in dedup.iterrows():
        fid = int(row['frame_id'])
        cs, cd = str(row.get('canonical_src','')), str(row.get('canonical_dst',''))
        if cs != 'None' and cd != 'None':
            pairs[fid] = (cs, cd)
    return pairs, selected

def run_grid(name, rolling_path, offline_path, ckpts_json_path=None):
    """Run full grid for one rolling output."""
    rolling = pd.read_parquet(rolling_path)
    offline = pd.read_parquet(offline_path)
    offline_pairs = load_pairs(offline, key='raw')
    off_ref = {f:p for f,p in offline_pairs.items() if FS<=f<=FE}
    
    # Timing from checkpoints JSON
    ckpts_data = {}
    if ckpts_json_path and Path(ckpts_json_path).exists():
        ckpts_data = json.loads(Path(ckpts_json_path).read_text())
    
    rows = []
    for cad in CADENCES:
        for delay in DELAYS:
            emit_pairs, sel = simulate(rolling, cad, delay)
            emit_f = {f:p for f,p in emit_pairs.items() if FS<=f<=FE}
            n, p, r, f1 = match(emit_f, off_ref)
            
            dedup_pairs, dsel = dedup_edges(rolling, cad)
            dedup_f = {f:p for f,p in dedup_pairs.items() if FS<=f<=FE}
            dn, dp, dr, df1 = match(dedup_f, off_ref)
            
            # Estimate timing from checkpoint data
            n_ckpts = len(sel)
            avg_ms = ckpts_data.get('avg_infer_ms', 0) if ckpts_data else 0
            max_ms = max([c.get('infer_ms',0) for c in ckpts_data.get('checkpoints',[])]) if ckpts_data else 0
            
            rows.append({
                'name': name, 'cadence': cad, 'delay': delay,
                'emitted_f1': round(f1,4), 'emitted_prec': round(p,4), 'emitted_rec': round(r,4),
                'emitted_pred': len(emit_f), 'emitted_matches': n,
                'dedup_f1': round(df1,4), 'dedup_prec': round(dp,4), 'dedup_rec': round(dr,4),
                'dedup_pred': len(dedup_f), 'dedup_matches': dn,
                'checkpoints': n_ckpts, 'avg_infer_ms': round(avg_ms,1), 'max_infer_ms': round(max_ms,1),
                'ref_edges': len(off_ref),
            })
    return rows

# ── Run grid on available data ──
all_rows = []

# Run 1: cad=10, window=500 (from rolling_windowed_v2)
# We need to re-run cad=10 or use available data
# Run 2: cad=25, window=500 (from current)
if Path('output/actions/rolling_online/partido_cad25/rolling_edges_per_frame.parquet').exists():
    rows = run_grid(
        'w500_cad25',
        'output/actions/rolling_online/partido_cad25/rolling_edges_per_frame.parquet',
        'output/actions/pathcrf_eval_cad25/tracks_edge_sequence.parquet',
        'output/actions/rolling_online/partido_cad25/runtime_checkpoints.json',
    )
    all_rows.extend(rows)
    print(f"cad=25: {len(rows)} grid entries")

# Print results
df = pd.DataFrame(all_rows)
if not df.empty:
    print("\n=== GRID SUMMARY ===\n")
    print(f"{'Name':<14s} {'Cad':>4s} {'Dly':>4s} {'EmitF1':>7s} {'DedupF1':>7s} {'Pred':>5s} {'Match':>5s} {'Ckpt':>5s} {'Avg_ms':>7s} {'Max_ms':>7s}")
    print("-" * 80)
    for _, r in df.iterrows():
        print(f"{r['name']:<14s} {r['cadence']:>4d} {r['delay']:>4d} {r['emitted_f1']:>7.4f} {r['dedup_f1']:>7.4f} {r['emitted_pred']:>5d} {r['emitted_matches']:>5d} {r['checkpoints']:>5d} {r['avg_infer_ms']:>7.0f} {r['max_infer_ms']:>7.0f}")
    
    csv_path = Path('output/actions/rolling_grid_eval.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nSaved to {csv_path}")
else:
    print("No data found.")
