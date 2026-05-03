"""Sweep tail combos on 200-frame smoke test."""
import subprocess, json, time, sys, pandas as pd, numpy as np
from pathlib import Path

combos = [
    (10, 25),
    (25, 75),
    (50, 50),
]
results = []

for overlap, tail in combos:
    # Write config
    config_path = Path('config.yaml')
    config = config_path.read_text()
    config = config.replace('incremental_tail_overlap_frames: 100', f'incremental_tail_overlap_frames: {overlap}')
    config = config.replace('incremental_recompute_tail_frames: 75', f'incremental_recompute_tail_frames: {tail}')
    config_path.write_text(config)
    
    label = f"tail_o{overlap}_t{tail}"
    print(f"\n=== {label} ===")
    
    t0 = time.time()
    r = subprocess.run([
        sys.executable, 'scripts/track.py', 'video_prueba_medio',
        '--skip-render-video',
        '--experiment-label', label,
        '--output-root', f'output/tracks_json/{label}',
    ], capture_output=True, text=True, timeout=300, cwd=Path(__file__).parent.parent)
    wall = time.time() - t0
    
    # Parse output
    infer_times = []
    for line in r.stdout.split('\n') + r.stderr.split('\n'):
        if 'tracking=' in line and 'infer_ms=' in line:
            infer_times.append(float(line.split('infer_ms=')[-1]))
    
    # Check artifacts
    art_dir = Path('output/actions/rolling_online/partido_medio')
    null = total = 0
    if (art_dir / 'emitted_edges.parquet').exists():
        e = pd.read_parquet(art_dir / 'emitted_edges.parquet')
        total = len(e)
        null = int((e['edge_src'].isna() | (e['edge_src']=='None')).sum())
    
    print(f'  Wall: {wall:.0f}s  Ckpts: {len(infer_times)}  Null: {null}/{total}')
    if infer_times:
        print(f'  Infer: avg={np.mean(infer_times):.0f}ms max={np.max(infer_times):.0f}ms p95={np.percentile(infer_times,95):.0f}ms')
    
    results.append({'overlap': overlap, 'tail': tail, 'wall_s': round(wall,1),
                    'ckpts': len(infer_times), 'null': null, 'total': total,
                    'avg_ms': round(np.mean(infer_times),0) if infer_times else 0,
                    'max_ms': round(np.max(infer_times),0) if infer_times else 0})

# Restore config
config = Path('config.yaml').read_text()
config = config.replace(f'incremental_tail_overlap_frames: {overlap}', 'incremental_tail_overlap_frames: 100')
config = config.replace(f'incremental_recompute_tail_frames: {tail}', 'incremental_recompute_tail_frames: 75')
Path('config.yaml').write_text(config)

print("\n=== SWEEP RESULTS ===")
print(f"{'Overlap':>7s} {'Tail':>5s} {'Wall':>6s} {'Ckpt':>5s} {'Null':>6s} {'Avg_ms':>7s} {'Max_ms':>7s}")
for r in results:
    print(f"{r['overlap']:>7d} {r['tail']:>5d} {r['wall_s']:>6.0f}s {r['ckpts']:>5d} {r['null']:>5d}/{r['total']:<4d} {r['avg_ms']:>7.0f} {r['max_ms']:>7.0f}")
