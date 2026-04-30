"""Autonomous optimization loop -- AGENTS.md-driven generic explorer."""

import os, time, random, copy
import numpy as np
from .evaluate import evaluate, score, acq_time
from .parser import get_defaults_for, apply_constraints, parse_instruction


def run_autonomous(instruction, num_experiments=100, output_dir='output',
                   mode='random', model_id=None, api_key=None):
    """Parse instruction, run optimization, return best params and sequence."""

    parsed = parse_instruction(instruction)
    seq_type = parsed['seq_type']
    defaults, param_meta = get_defaults_for(seq_type)
    constraints = parsed['constraints']

    # Determine which params are fixed by instruction vs. free to explore
    fixed_keys = set(constraints.keys())
    editable_meta = {k: v for k, v in param_meta.items() if k not in fixed_keys}

    params = apply_constraints(defaults, constraints)

    os.makedirs(output_dir, exist_ok=True)

    print('=' * 60)
    print(f'autoresearch-MRsequence -- TSE Sequence Optimization')
    print('=' * 60)
    print(f'Instruction: {instruction[:80]}...')
    print(f'Params: TE={params.get("te",0)*1000:.0f}ms TR={params.get("tr",0)*1000:.0f}ms'
          f' Matrix={params.get("n_x",128)}x{params.get("n_y",128)}')
    print(f'Fixed by instruction: {sorted(fixed_keys) if fixed_keys else "(none)"}')
    print(f'Explorable: {sorted(editable_meta.keys())}')
    print(f'Target: {num_experiments} experiments')
    print('-' * 60)

    # Baseline
    metrics = evaluate(params, output_dir, exp_id=1)
    acq_t = acq_time(params)
    base_mae = metrics['mae_total']
    base_sar = max(metrics.get('sar_estimate', 0.001), 0.0001)
    base_time = max(acq_t, 1.0)
    base_score = score(base_mae, base_sar, base_time, base_mae, base_sar, base_time)
    metrics['acq_time_s'] = acq_t
    metrics['score'] = base_score
    _print_report(params, seq_type, metrics)

    best_score = base_score; best_mae = base_mae
    best_params = dict(params)
    if 'rf_flip_angles' in best_params:
        best_params['rf_flip_angles'] = list(best_params['rf_flip_angles'])

    # TSV
    tsv = os.path.join(output_dir, 'results.tsv')
    with open(tsv, 'w') as f:
        f.write('exp\tmae_total\tmae_gm\tmae_wm\tmae_csf\tscore\tbest_score\t'
                'snr\tcnr\tpns_sar\tacq_time\tturbo\tmatrix\tte_ms\ttr_ms\t'
                'timing\tstatus\tdesc\n')
    _log(tsv, 1, params, metrics, best_mae, best_score, seq_type)

    # Initialize LLM agent if in LLM mode
    llm_agent = None
    llm_stats_path = None
    history = []
    if mode == 'llm' and model_id and api_key:
        from .llm_agent import LLMAgent
        import os as _os
        program_md_path = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)), 'AGENTS.md')
        with open(program_md_path) as f:
            program_md = f.read()
        llm_agent = LLMAgent(model_id, api_key, program_md, editable_meta, params)
        llm_stats_path = os.path.join(output_dir, 'llm_stats.tsv')
        with open(llm_stats_path, 'w') as f:
            f.write('exp\tmodel\ttokens_in\ttokens_out\treasoning\tstatus\n')
        print(f'LLM Agent: {model_id}')

    t_start = time.time()
    no_improve = 0
    try:
        for exp in range(2, num_experiments + 1):
            if llm_agent:
                p, llm_stats = llm_agent.propose(best_params, history, exp)
                score_val = best_score
                with open(llm_stats_path, 'a') as f:
                    f.write(f'{exp}\t{model_id}\t{llm_stats.get("tokens_in",0)}\t'
                            f'{llm_stats.get("tokens_out",0)}\t{llm_stats.get("reasoning",0)}\t'
                            f'{llm_stats.get("error","ok")}\n')
            else:
                p = _propose(best_params, editable_meta, exp / num_experiments)
            desc = _describe(p, editable_meta, seq_type)

            try:
                metrics = evaluate(p, output_dir, exp_id=exp, fast_mode=True)
            except Exception as e:
                metrics = {'status': 'crash', 'error': str(e), 'mae_total': 0, 'score': 999}

            acq_t = acq_time(p)
            if constraints.get('max_acq_time', float('inf')) < acq_t:
                metrics['score'] = 999

            s = score(metrics.get('mae_total', 0), metrics.get('sar_estimate', 0.01),
                      acq_t, base_mae, base_sar, base_time)
            metrics['acq_time_s'] = acq_t; metrics['score'] = s
            _log(tsv, exp, p, metrics, best_mae, best_score, seq_type)

            # Track history for LLM agent
            history.append({'exp': exp, 'mae': metrics.get('mae_total', 0),
                           'score': s, 'status': metrics.get('status', '?'),
                           'params': dict(p)})
            if len(history) > 20:
                history = history[-15:]

            if metrics['status'] == 'crash':
                continue

            if 0 < s < best_score:
                best_score = s; best_mae = metrics['mae_total']
                best_params = dict(p)
                if 'rf_flip_angles' in best_params:
                    best_params['rf_flip_angles'] = list(best_params['rf_flip_angles'])
                evaluate(best_params, output_dir, exp_id=exp, fast_mode=False)
                print(f'  KEEP #{exp}: score={s:.4f} MAE={metrics["mae_total"]:.4f}')
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= 15:
                print(f'  Converged: no improvement for 15 experiments, stopping at #{exp}')
                break

            if exp % 10 == 0 or exp == num_experiments:
                print(f'  [{exp}/{num_experiments}] Best: {best_score:.4f}')
                _save_progress(tsv, os.path.join(output_dir, 'progress.png'), best_params)
    except KeyboardInterrupt:
        print(f'\nInterrupted at experiment #{exp}. Generating final outputs...')

    # Final outputs
    from .sequences import SEQ_BUILDERS
    seq, _, _, _ = SEQ_BUILDERS[seq_type](**{k: v for k, v in best_params.items() if k != 'noise_snr'})
    seq.write(os.path.join(output_dir, 'best_sequence.seq'))
    evaluate(best_params, output_dir, exp_id=0, fast_mode=False)
    _save_progress(tsv, os.path.join(output_dir, 'progress.png'), best_params)

    # Generate waveform + k-space analysis + report
    try:
        from .report import generate_all
        import MRzeroCore as mr0
        from .evaluate import load_phantom
        phantom = load_phantom(size=(best_params.get('n_x', 128), best_params.get('n_y', 128)))
        signal_r, kspace_r = mr0.util.simulate(seq, phantom=phantom, accuracy=0.003)
        generate_all(output_dir, best_params, seq, signal_r, kspace_r, tsv)
    except Exception as e:
        print(f'Report generation skipped: {e}')

    elapsed = time.time() - t_start
    print(f'\nDone! {num_experiments} exps in {elapsed:.0f}s ({elapsed/60:.1f}m)')
    print(f'Best: MAE={best_mae:.4f} Score={best_score:.4f}')
    if llm_agent:
        ci = llm_agent.get_cost_info()
        print(f'LLM: {ci["calls"]} calls, {ci["tokens_in"]}+{ci["tokens_out"]} tokens, ${ci["cost"]:.4f}')
    print(f'Outputs: {output_dir}/')
    return best_params


# ---------------------------------------------------------------------------
# Generic exploration engine (AGENTS.md-driven, no type-specific rules)
# ---------------------------------------------------------------------------

def _propose(best_params, editable_meta, progress):
    """Generate candidate parameter set based on metadata.
    Varies 1-3 parameters per experiment for meaningful exploration."""
    exploration = 0.5 * (1 - progress) + 0.1
    p = copy.deepcopy(best_params)
    if not editable_meta:
        return p

    # Vary 1-3 parameters (more exploration early, exploit late)
    n_change = min(len(editable_meta), random.randint(1, 3))
    keys = random.sample(list(editable_meta.keys()), n_change)

    for key in keys:
        meta = editable_meta[key]
        if random.random() < exploration:
            _explore(p, key, meta)
        else:
            _perturb(p, key, meta)

    _resolve_deps(p, editable_meta)
    return p


def _explore(p, key, meta):
    """Randomly sample parameter from its defined range."""
    t = meta['type']
    if t == 'int':
        if 'valid' in meta:
            p[key] = random.choice(meta['valid'])
        else:
            lo, hi = meta['range']
            p[key] = random.randint(lo, hi)
    elif t == 'float':
        lo, hi = meta['range']
        p[key] = round(random.uniform(lo, hi), 4)
    elif t == 'choice':
        p[key] = random.choice(meta['choices'])
    elif t == 'list':
        lo, hi = meta['range']
        n = len(p[key])
        p[key] = [round(random.uniform(lo, hi)) for _ in range(n)]


def _perturb(p, key, meta):
    """Perturb parameter around current value."""
    t = meta['type']
    if t == 'int':
        if 'valid' in meta:
            p[key] = random.choice(meta['valid'])
        else:
            lo, hi = meta['range']
            delta = random.randint(-2, 2)
            p[key] = max(lo, min(hi, p[key] + delta))
    elif t == 'float':
        lo, hi = meta['range']
        span = abs(hi - lo)
        delta = random.uniform(-0.05 * span, 0.05 * span)
        p[key] = round(max(lo, min(hi, p[key] + delta)), 4)
    elif t == 'choice':
        others = [c for c in meta['choices'] if c != p[key]]
        if others:
            p[key] = random.choice(others)
    elif t == 'list':
        lo, hi = meta['range']
        mag = meta.get('perturb_mag', 15)
        p[key] = list(p[key])
        n = len(p[key])
        for i in random.sample(range(n), min(max(1, n // 3), n)):
            delta = random.randint(-mag, mag)
            p[key][i] = max(lo, min(hi, round(p[key][i] + delta)))


def _resolve_deps(p, editable_meta):
    """Resolve parameter dependencies (e.g., n_echo -> rf_flip_angles length)."""
    for key, meta in editable_meta.items():
        dep_key = meta.get('list_length_key')
        if dep_key and dep_key in editable_meta:
            target_len = p.get(dep_key, len(p.get(key, [])))
            current = p.get(key)
            if isinstance(current, list) and len(current) != target_len:
                lo, hi = meta['range']
                p[key] = [round(random.uniform(lo, hi)) for _ in range(target_len)]


# ---------------------------------------------------------------------------
# Logging and reporting
# ---------------------------------------------------------------------------

def _describe(p, meta, seq_type):
    enc = p.get('encoding', 'linear')
    return f'flips={p.get("rf_flip_angles",[])} enc={enc} turbo={p.get("n_echo","?")}'


def _log(path, exp_id, params, metrics, best_mae, best_score, seq_type):
    per = metrics.get('mae_per_tissue', {})
    te_ms = params.get('te', 0.08) * 1000
    tr_ms = params.get('tr', 3.0) * 1000
    desc = _describe(params, {}, seq_type)
    with open(path, 'a') as f:
        f.write(f'{exp_id}\t{metrics.get("mae_total",0):.6f}\t'
                f'{per.get("GM",0):.4f}\t{per.get("WM",0):.4f}\t{per.get("CSF",0):.4f}\t'
                f'{metrics.get("score",0):.4f}\t{best_score:.4f}\t'
                f'{metrics.get("snr",0):.1f}\t{metrics.get("cnr_gm_wm",0):.2f}\t'
                f'{metrics.get("sar_estimate",0):.4f}\t{metrics.get("acq_time_s",0):.0f}\t'
                f'{params.get("n_echo", params.get("n_y",0))}\t{params.get("n_x",128)}x{params.get("n_y",128)}\t'
                f'{te_ms:.0f}\t{tr_ms:.0f}\t'
                f'{metrics.get("timing_check","N/A")}\t{metrics.get("status","?")}\t{desc}\n')


def _print_report(params, seq_type, metrics):
    print(f'\n--- MAE={metrics["mae_total"]:.4f} SAR={metrics.get("sar_estimate",0):.4f} '
          f'Time={metrics.get("acq_time_s",0):.0f}s ---\n')


def _save_progress(tsv_path, out_path, best_params):
    """Generate 2x2 progress panel: Score descent + MAE vs SAR + Convergence + Best params."""
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if not os.path.exists(tsv_path): return
    with open(tsv_path) as f:
        lines = f.readlines()
    if len(lines) < 2: return
    data = []
    improvements = []
    curr_best = float('inf')
    for line in lines[1:]:
        parts = line.strip().split('\t')
        if len(parts) < 7: continue
        try:
            d = {'exp': int(parts[0]), 'mae': float(parts[1]), 'score': float(parts[4]),
                 'sar': float(parts[8]) if len(parts) > 8 else 0,
                 'time': float(parts[9]) if len(parts) > 9 else 0,
                 'turbo': int(parts[10]) if len(parts) > 10 else 0,
                 'status': parts[-2] if len(parts) > 1 else 'ok'}
            data.append(d)
            if d['score'] > 0 and d['score'] < curr_best:
                curr_best = d['score']; improvements.append((d['exp'], d['score']))
        except (ValueError, IndexError): pass
    if not data: return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('autoresearch-MRsequence -- Sequence Optimization Progress', fontsize=13, fontweight='bold')

    # Panel 1: Score descent
    ax = axes[0, 0]
    valid = [(d['exp'], d['score']) for d in data if d['score'] > 0 and d['status'] != 'crash']
    if valid:
        ax.scatter(*zip(*valid), c='#e0e0e0', s=10, zorder=1, label='Experiments')
    if improvements:
        ix, iy = zip(*improvements)
        ax.step(ix, iy, 'b-', lw=2.5, where='post', zorder=4, label=f'Best={curr_best:.4f}')
        ax.scatter(ix, iy, c='#2ecc71', s=50, zorder=5, edgecolors='darkgreen')
    ax.axhline(y=data[0]['score'], color='orange', ls='--', alpha=0.5, label='Baseline')
    ax.set_xlabel('Experiment #'); ax.set_ylabel('Score (lower=better)')
    ax.set_title(f'Score Descent ({len(improvements)} improvements)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

    # Panel 2: MAE vs SAR tradeoff
    ax = axes[0, 1]
    valid_d = [d for d in data if d['status'] != 'crash' and d['mae'] > 0]
    if valid_d:
        ax.scatter([d['sar'] for d in valid_d], [d['mae'] for d in valid_d],
                   s=15, c='#bdc3c7', alpha=0.5, zorder=1)
        best_d = min(valid_d, key=lambda d: d['score'])
        ax.scatter([best_d['sar']], [best_d['mae']], s=120, c='red', zorder=5, marker='*')
        ax.annotate(f'Best\nMAE={best_d["mae"]:.3f}', (best_d['sar'], best_d['mae']),
                    xytext=(10, 10), textcoords='offset points', fontsize=8)
    ax.set_xlabel('SAR Estimate'); ax.set_ylabel('MAE Total')
    ax.set_title('MAE vs SAR Tradeoff')
    ax.grid(True, alpha=0.2)

    # Panel 3: Convergence (running best)
    ax = axes[1, 0]
    running = [float('inf')]
    for d in data:
        s = d['score'] if d['score'] > 0 else running[-1]
        running.append(min(running[-1], s))
    running = running[1:]
    ax.plot(range(1, len(running)+1), running, 'b-', alpha=0.8)
    ax.fill_between(range(1, len(running)+1), running, alpha=0.1, color='blue')
    ax.set_xlabel('Experiment #'); ax.set_ylabel('Best Score So Far')
    ax.set_title('Convergence'); ax.grid(True, alpha=0.2)

    # Panel 4: Best parameters text
    ax = axes[1, 1]; ax.axis('off')
    if best_params:
        flips = best_params.get('rf_flip_angles', best_params.get('flip_angle', 'N/A'))
        flip_str = str(flips[:4])[:-1] + '...]' if isinstance(flips, list) and len(flips) > 4 else str(flips)
        text = (f'BEST PARAMETERS\n{"="*30}\n'
                f'TE: {best_params["te"]*1000:.0f} ms\n'
                f'TR: {best_params["tr"]*1000:.0f} ms\n'
                f'Turbo: {best_params.get("n_echo","?")}\n'
                f'Matrix: {best_params["n_x"]}x{best_params["n_y"]}\n'
                f'FOV: {best_params["fov"]} m\n'
                f'Encoding: {best_params.get("encoding","linear")}\n'
                f'Flips: {flip_str}\n'
                f'\nw_MAE=0.5  w_SAR=0.3  w_time=0.2')
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=9,
                fontfamily='monospace', verticalalignment='top')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
