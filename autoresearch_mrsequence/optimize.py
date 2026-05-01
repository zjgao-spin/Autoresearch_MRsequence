"""Autonomous optimization loop -- AGENTS.md-driven generic explorer."""

import os, time, random, copy
import numpy as np
from .evaluate import evaluate, score, acq_time
from .parser import get_defaults_for, apply_constraints, parse_instruction


def run_autonomous(instruction, num_experiments=100, output_dir='output',
                   mode='random', model_id=None, api_key=None, seq_type=None):
    """Parse instruction, run optimization, return best params and sequence."""

    parsed = parse_instruction(instruction)
    if seq_type is None:
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
    metrics = evaluate(params, output_dir, exp_id=1, seq_type=seq_type)
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
                metrics = evaluate(p, output_dir, exp_id=exp, fast_mode=True, seq_type=seq_type)
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

            # Live panel — every experiment with current params
            info = {'exp': exp, 'mae': metrics.get('mae_total', 0),
                    'score': s, 'sar': metrics.get('sar_estimate', 0),
                    'time': metrics.get('acq_time_s', 0),
                    'encoding': p.get('encoding', '?'), 'turbo': p.get('n_echo', '?')}
            _save_live_panel(output_dir, p, seq_type, tsv, exp, info)

            if 0 < s < best_score:
                best_score = s; best_mae = metrics['mae_total']
                best_params = dict(p)
                if 'rf_flip_angles' in best_params:
                    best_params['rf_flip_angles'] = list(best_params['rf_flip_angles'])
                evaluate(best_params, output_dir, exp_id=exp, fast_mode=False, seq_type=seq_type)
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
    evaluate(best_params, output_dir, exp_id=0, fast_mode=False, seq_type=seq_type)
    _save_progress(tsv, os.path.join(output_dir, 'progress.png'), best_params)
    info_final = {'exp': num_experiments, 'mae': best_mae, 'score': best_score,
                  'sar': 0, 'time': 0,
                  'encoding': best_params.get('encoding', '?'),
                  'turbo': best_params.get('n_echo', '?')}
    _save_live_panel(output_dir, best_params, seq_type, tsv, num_experiments, info_final)

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
# Live 4-panel figure: sim+target | waveform | kspace | MAE descent
# ---------------------------------------------------------------------------

def _save_live_panel(output_dir, best_params, seq_type, tsv_path, exp_num, current_info=None):
    """Generate a real-time 4-panel overview of current best sequence."""
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as _np
    from io import BytesIO
    from .sequences import SEQ_BUILDERS
    from .evaluate import load_phantom, theoretical_target
    import MRzeroCore as mr0
    import os as _os
    import warnings

    # ---- ensure clean matplotlib state ----
    plt.close('all')

    # ---- build + simulate ----
    build_params = {k: v for k, v in best_params.items() if k != 'noise_snr'}
    try:
        seq, _, _, _ = SEQ_BUILDERS[seq_type](**build_params)
    except Exception as e:
        print(f'  Live panel: seq build failed ({e})')
        return

    phantom = load_phantom(size=(best_params.get('n_x', 128), best_params.get('n_y', 128)))
    try:
        signal, kspace = mr0.util.simulate(seq, phantom=phantom, accuracy=0.005)
    except Exception as e:
        print(f'  Live panel: simulate failed ({e})')
        return

    n_x, n_y = best_params.get('n_x', 128), best_params.get('n_y', 128)
    fov = best_params.get('fov', 0.20)
    img = mr0.reco_adjoint(signal, kspace, resolution=(n_x, n_y, 1), FOV=(fov, fov, 1.0))
    img_np = img.abs().squeeze().cpu().numpy()

    TE = best_params.get('te', 0.08)
    TR = best_params.get('tr', 3.0)
    target_tensor, _ = theoretical_target(phantom, TE, TR)
    target_np = target_tensor.cpu().numpy()

    # ---- render each panel independently into BytesIO buffers ----
    panels = {}  # name -> BytesIO

    # Panel 1: Simulated vs Target (side-by-side, with scaling)
    fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(8, 5))

    # Least-squares scale simulated image to match target magnitude
    mask_tissue = target_np > target_np.max() * 0.01
    if mask_tissue.any():
        num = (target_np[mask_tissue] * img_np[mask_tissue]).sum()
        den = (img_np[mask_tissue] * img_np[mask_tissue]).sum()
        scale = num / max(den, 1e-8)
        img_scaled = img_np * scale
    else:
        img_scaled = img_np

    vmin = min(target_np[mask_tissue].min() if mask_tissue.any() else 0,
               img_scaled[mask_tissue].min() if mask_tissue.any() else 0)
    vmax = max(target_np[mask_tissue].max() if mask_tissue.any() else 1,
               img_scaled[mask_tissue].max() if mask_tissue.any() else 1)

    ax1a.imshow(img_scaled, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    ax1a.set_title('Simulated', fontsize=11, fontweight='bold')
    ax1a.axis('off')
    ax1b.imshow(target_np, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    te_ms, tr_ms = TE * 1000, TR * 1000
    ax1b.set_title(f'Target  TE={te_ms:.0f}ms TR={tr_ms:.0f}ms', fontsize=11, fontweight='bold')
    ax1b.axis('off')
    fig1.tight_layout()
    buf1 = BytesIO()
    fig1.savefig(buf1, format='png', dpi=120, facecolor='white', pad_inches=0.3)
    buf1.seek(0); panels['sim'] = buf1
    plt.close(fig1)

    # Panel 2: Sequence Waveform (via PyPulseq seq.plot, captured as image)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='FigureCanvasAgg')
            warnings.filterwarnings('ignore', message='This figure includes Axes')
            seq.plot()
        fig_w = plt.gcf()
        fig_w.set_size_inches(12, 5)
        buf2 = BytesIO()
        fig_w.savefig(buf2, format='png', dpi=90, facecolor='white', bbox_inches='tight')
        buf2.seek(0); panels['wave'] = buf2
        plt.close(fig_w)
    except Exception:
        # fallback: empty panel
        fig_fb, ax_fb = plt.subplots(figsize=(8, 5))
        ax_fb.text(0.5, 0.5, 'Waveform unavailable', ha='center', va='center',
                   fontsize=12, transform=ax_fb.transAxes)
        ax_fb.axis('off')
        fig_fb.tight_layout()
        buf2 = BytesIO()
        fig_fb.savefig(buf2, format='png', dpi=120, facecolor='white', pad_inches=0.3)
        buf2.seek(0); panels['wave'] = buf2
        plt.close(fig_fb)

    # Panel 3: K-Space PE Trajectory
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ksp_np = kspace.cpu().numpy() if hasattr(kspace, 'cpu') else kspace
    ky = ksp_np[:, 1]
    n_echo = best_params.get('n_echo', 8)
    n_pe = best_params.get('n_y', 128)
    encoding = best_params.get('encoding', 'linear')
    n_ex = int(_np.floor(n_pe / n_echo))
    n_total = n_ex * n_echo
    rmid = n_x // 2
    echo_ky = _np.array([ky[e * n_x + rmid] for e in range(n_total)])
    grid = echo_ky[:n_total].reshape(n_ex, n_echo).T
    for exc in range(min(n_ex, 12)):
        a = 0.9 if exc < 6 else 0.2
        ax3.plot(range(n_echo), grid[:, exc], 'o-', ms=4, lw=1.2, alpha=a)
    ax3.axhline(0, color='red', ls='--', alpha=0.5, lw=1.0)
    ax3.set_xlabel('Echo #', fontsize=10); ax3.set_ylabel('ky (1/m)', fontsize=10)
    ax3.set_title(f'K-Space PE Order  ({encoding}, turbo={n_echo})', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.15)
    fig3.tight_layout()
    buf3 = BytesIO()
    fig3.savefig(buf3, format='png', dpi=120, facecolor='white', pad_inches=0.3)
    buf3.seek(0); panels['ksp'] = buf3
    plt.close(fig3)

    # Panel 4: MAE Descent
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    mae_data = []
    if _os.path.exists(tsv_path):
        with open(tsv_path) as f:
            for li, line in enumerate(f):
                if li == 0: continue
                parts = line.strip().split('\t')
                if len(parts) < 3: continue
                try:
                    m = float(parts[1])
                    if m > 0: mae_data.append(m)
                except (ValueError, IndexError): pass
    if mae_data:
        ax4.scatter(range(1, len(mae_data) + 1), mae_data,
                    s=10, c='#cccccc', zorder=1, label='Experiments')
        running, best_m = [], float('inf')
        for m in mae_data:
            best_m = min(best_m, m)
            running.append(best_m)
        ax4.step(range(1, len(running) + 1), running, 'b-', lw=2.5,
                 where='post', zorder=3, label=f'Best={running[-1]:.4f}')
        ax4.fill_between(range(1, len(running) + 1), running, alpha=0.06, color='blue')
        ax4.legend(fontsize=8, loc='upper right')
    ax4.set_xlabel('Experiment #', fontsize=10); ax4.set_ylabel('MAE Total', fontsize=10)
    ax4.set_title('MAE Descent', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.15)

    # Metrics overlay
    if current_info:
        info_text = (
            f"Exp #{current_info.get('exp', exp_num)}  "
            f"Score: {current_info.get('score', 0):.4f}\n"
            f"MAE: {current_info.get('mae', 0):.4f}  "
            f"SAR: {current_info.get('sar', 0):.4f}  "
            f"Time: {current_info.get('time', 0):.0f}s\n"
            f"encoding={current_info.get('encoding', '?')}  "
            f"turbo={current_info.get('turbo', '?')}"
        )
        ax4.text(0.02, 0.97, info_text, transform=ax4.transAxes, fontsize=6.5,
                 fontfamily='monospace', verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.85))

    fig4.tight_layout()
    buf4 = BytesIO()
    fig4.savefig(buf4, format='png', dpi=120, facecolor='white', pad_inches=0.3)
    buf4.seek(0); panels['mae'] = buf4
    plt.close(fig4)

    # ---- compose 2x2 grid ----
    fig_all = plt.figure(figsize=(21, 15))
    fig_all.patch.set_facecolor('white')
    fig_all.suptitle(f'autoresearch-MRsequence — Current Best  (Exp #{exp_num})',
                     fontsize=15, fontweight='bold', y=0.99)

    names = ['sim', 'wave', 'ksp', 'mae']
    titles = ['Simulated vs Target', 'Sequence Waveform',
              'K-Space PE Trajectory', 'MAE Descent']
    for idx, (name, title) in enumerate(zip(names, titles)):
        ax = fig_all.add_subplot(2, 2, idx + 1)
        ax.axis('off')
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        if name in panels:
            panels[name].seek(0)
            panel_img = plt.imread(panels[name])
            ax.imshow(panel_img)
        panels[name].close()  # free memory

    out_path = _os.path.join(output_dir, 'live_panel.png')
    fig_all.savefig(out_path, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close(fig_all)
    plt.close('all')

    # Save numbered frame for animations
    import shutil
    frame_dir = _os.path.join(output_dir, 'frames')
    _os.makedirs(frame_dir, exist_ok=True)
    shutil.copy(out_path, _os.path.join(frame_dir, f'live_{exp_num:04d}.png'))

    print(f'  Live panel updated: {out_path}')


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
        cur = p.get(key)
        if cur is None or not isinstance(cur, list):
            length_key = meta.get('list_length_key', '')
            if length_key and length_key in p:
                n = p[length_key]
            else:
                n = 8
            p[key] = [round(random.uniform(lo, hi)) for _ in range(n)]
        else:
            n = len(cur)
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
        cur = p.get(key)
        if cur is None or not isinstance(cur, list):
            # Fallback: initialize from list_length_key or default length
            length_key = meta.get('list_length_key', '')
            if length_key and length_key in p:
                n = p[length_key]
            else:
                n = 8
            p[key] = [round(random.uniform(lo, hi)) for _ in range(n)]
        else:
            p[key] = list(cur)
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
# Agent workspace ? python -m autoresearch_mrsequence.optimize

def _write_tsv_agent(path, exp_id, mae_total):
    import os as _os
    existed = _os.path.exists(path)
    with open(path, 'a') as f:
        if not existed:
            f.write('exp\tmae_total\tscore\n')
        f.write(f'{exp_id}\t{mae_total:.6f}\t0\n')


if __name__ == '__main__':
    import os as _os, json

    output_dir = 'output'; seq_type = 'tse'
    _os.makedirs(output_dir, exist_ok=True)
    tsv = _os.path.join(output_dir, 'results.tsv')
    state_path = _os.path.join(output_dir, 'state.json')

    if _os.path.exists(state_path):
        with open(state_path) as f:
            state = json.load(f)
        best_params = state['best_params']
        best_score = state['best_score']
        baseline_mae = state['baseline_mae']
        baseline_sar = state['baseline_sar']
        baseline_time = state['baseline_time']
        last_exp = state['exp']
    else:
        best_params = {
            'fov': 0.20, 'n_x': 128, 'n_y': 128, 'n_echo': 8,
            'rf_flip_angles': [180]*8, 'slice_thickness': 5e-3,
            'te': 0.08, 'tr': 3.0, 'fsp_r': 1.0, 'fsp_s': 0.5, 'encoding': 'linear',
        }
        best_score = float('inf')
        baseline_mae = baseline_sar = baseline_time = 1.0
        last_exp = 0

    if last_exp == 0:
        print('=' * 50)
        print('Running baseline (exp_id=1) ...')
        p = dict(best_params)
        m = evaluate(p, output_dir, exp_id=1)
        baseline_mae = m['mae_total']
        baseline_sar = max(m.get('sar_estimate', 0.01), 0.0001)
        baseline_time = max(m.get('acq_time_s', 0), 1.0)
        best_score = score(baseline_mae, baseline_sar, baseline_time,
                           baseline_mae, baseline_sar, baseline_time)
        print(f'Baseline  MAE={baseline_mae:.4f}  Score={best_score:.4f}')
        _write_tsv_agent(tsv, 1, baseline_mae)
        last_exp = 1
        info_bl = {'exp': 1, 'mae': baseline_mae, 'score': best_score,
                   'sar': baseline_sar, 'time': baseline_time,
                   'encoding': p.get('encoding', '?'), 'turbo': p.get('n_echo', '?')}
        _save_live_panel(output_dir, p, seq_type, tsv, 1, info_bl)
        with open(state_path, 'w') as f:
            json.dump({'best_params': best_params, 'best_score': best_score,
                       'baseline_mae': baseline_mae, 'baseline_sar': baseline_sar,
                       'baseline_time': baseline_time, 'exp': 1}, f, default=str)

    if last_exp > 0:
        evaluate(best_params, output_dir, exp_id=1, fast_mode=True)

    # ====================================================================
    # EXPERIMENTS ? EDIT THE LIST BELOW, THEN RUN AGAIN
    # ====================================================================
    # Each dict overrides entries in best_params.
    # You may vary: encoding, n_echo, rf_flip_angles, fsp_r, fsp_s
    # Dependency: rf_flip_angles length MUST equal n_echo.

    experiments = [
        # ===== ADD YOUR EXPERIMENTS HERE =====
        {'encoding': 'centric'},
        {'encoding': 'centric', 'n_echo': 16, 'rf_flip_angles': [160,150,140,130,120,110,100,90,80,70,60,50,40,30,20,10]},
        {'n_echo': 4, 'rf_flip_angles': [150,140,130,120]},
    ]
    # ====================================================================
    # END EXPERIMENTS
    # ====================================================================

    if not experiments:
        print('experiments list is empty ? add parameter overrides and re-run.')
        enc = best_params.get("encoding"); tur = best_params.get("n_echo")
        print(f'Current best: Score={best_score:.4f}  encoding={enc}, turbo={tur}')
    else:
        improvements = 0
        start_exp = last_exp
        for i, override in enumerate(experiments):
            exp_id = start_exp + 1 + i
            params = dict(best_params)
            params.update(override)
            if isinstance(params.get('rf_flip_angles'), list):
                n_e = params.get('n_echo', 8)
                flips = params['rf_flip_angles']
                if len(flips) != n_e:
                    if len(flips) < n_e:
                        flips = flips + [flips[-1]] * (n_e - len(flips))
                    else:
                        flips = flips[:n_e]
                    params['rf_flip_angles'] = flips
            m = evaluate(params, output_dir, exp_id=exp_id, fast_mode=True)
            s = score(m.get('mae_total', baseline_mae), m.get('sar_estimate', baseline_sar),
                      m.get('acq_time_s', baseline_time), baseline_mae, baseline_sar, baseline_time)
            mae_val = m.get('mae_total', 0)
            print(f'Exp {exp_id:3d}  MAE={mae_val:.4f}  Score={s:.4f}')
            _write_tsv_agent(tsv, exp_id, mae_val)
            info = {'exp': exp_id, 'mae': mae_val, 'score': s,
                    'sar': m.get('sar_estimate', 0), 'time': m.get('acq_time_s', 0),
                    'encoding': params.get('encoding', '?'), 'turbo': params.get('n_echo', '?')}
            _save_live_panel(output_dir, params, seq_type, tsv, exp_id, info)
            if 0 < s < best_score:
                best_score = s
                best_params = dict(params)
                if isinstance(best_params.get('rf_flip_angles'), list):
                    best_params['rf_flip_angles'] = list(best_params['rf_flip_angles'])
                evaluate(best_params, output_dir, exp_id=exp_id, fast_mode=False)
                print(f'  >>> KEEP #{exp_id}')
                improvements += 1
            last_exp = exp_id
        with open(state_path, 'w') as f:
            json.dump({'best_params': best_params, 'best_score': best_score,
                       'baseline_mae': baseline_mae, 'baseline_sar': baseline_sar,
                       'baseline_time': baseline_time, 'exp': last_exp}, f, default=str)
        print(f'\nBatch done.  {improvements} improvement(s) in this run ({len(experiments)} exps).')

    from .sequences import SEQ_BUILDERS
    build_params = {k: v for k, v in best_params.items() if k != 'noise_snr'}
    seq, ok, _, _ = SEQ_BUILDERS[seq_type](**build_params)
    seq.write(f'{output_dir}/best_sequence.seq')
    print(f'Best Score={best_score:.4f}  Total exps: {last_exp}')
    print(f'Saved: {output_dir}/best_sequence.seq')
    print(f'Live panel: {output_dir}/live_panel.png')
    print('=' * 50)
