"""
Fixed oracle — DO NOT MODIFY. Analogous to prepare.py in karpathy/autoresearch.
Refs: karpathy/autoresearch, MRzero-Core, PyPulseq,
      Agent4MR (Zaiss et al., arXiv:2604.13282)
"""

import os, time, gc
import numpy as np
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import MRzeroCore as mr0


_phantom_cache = None
_baseline_target = None
_baseline_masks = None

def load_phantom(size=(128, 128)):
    global _phantom_cache
    if _phantom_cache is None:
        p = mr0.util.load_phantom(size=size)
        p.PD = p.PD.to(torch.float32)
        p.T1 = p.T1.to(torch.float32)
        p.T2 = p.T2.to(torch.float32)
        _phantom_cache = p
    return _phantom_cache


def theoretical_target(phantom, TE, TR):
    """Compute per-voxel theoretical TSE signal.
    S = PD * (1 - e^{-TR/T1}) * e^{-TE/T2}
    """
    PD = phantom.PD.squeeze(-1)
    T1 = phantom.T1.squeeze(-1).clamp(1e-6)
    T2 = phantom.T2.squeeze(-1).clamp(1e-6)
    S = PD * (1 - torch.exp(-TR / T1)) * torch.exp(-TE / T2)

    # Tissue masks
    mask = PD > 0.01
    csf = mask & (T2 > 0.2) & (T1 > 2.0)
    gm = mask & (PD > 0.3) & ~csf & (T1 > 0.5)
    wm = mask & ~csf & ~gm & (PD > 0.1)
    return S, {'CSF': csf, 'GM': gm, 'WM': wm, 'all': mask}


def compute_mae(img, target, masks):
    """Global least-squares scaled MAE."""
    br = masks['all']
    num = (target[br] * img[br]).sum()
    den = (img[br] * img[br]).sum()
    scale = num / max(den, 1e-8)
    scaled = img * scale
    diff = (scaled - target).abs()
    total = diff[br].mean().item()
    per = {}
    for k, m in masks.items():
        if k != 'all' and m.sum() > 0:
            per[k] = diff[m].mean().item()
    return total, per, scaled, scale


def acq_time(params):
    """Estimate scan time per slice (seconds)."""
    n_y = params['n_y']; n_echo = params['n_echo']
    n_ex = int(max(1, np.floor(n_y / n_echo)))
    return (n_ex + 1) * params['tr']


def score(mae, sar, acq_t, baseline_mae=1, baseline_sar=0.01, baseline_time=1):
    """Composite score: 0.5*MAE + 0.3*SAR + 0.2*Time (normalized)."""
    return (0.5 * mae / max(baseline_mae, 0.001) +
            0.3 * sar / max(baseline_sar, 0.001) +
            0.2 * acq_t / max(baseline_time, 0.001))


def evaluate(params, output_dir='output', exp_id=0, fast_mode=False):
    """Simulate sequence, reconstruct, compute metrics."""
    from .sequences import SEQ_BUILDERS
    builder = SEQ_BUILDERS.get('tse')
    if builder is None:
        return {'status': 'crash', 'error': 'Unknown sequence type',
                'mae_total': 0, 'score': 999}

    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()

    phantom = load_phantom(size=(params.get('n_x', 128), params.get('n_y', 128)))
    TE = params.get('te', 80e-3)
    TR = params.get('tr', 3000e-3)
    fov = params.get('fov', 0.20)

    # Build
    noise_snr = params.pop('noise_snr', 0)
    seq, ok, _, tr_delay = builder(**params)
    params['noise_snr'] = noise_snr  # restore
    # Simulate
    signal, kspace = mr0.util.simulate(seq, phantom=phantom, accuracy=0.005 if fast_mode else 0.001)

    # Inject Gaussian k-space noise (SNR relative to k-space center magnitude)
    if noise_snr and noise_snr > 0:
        ks_max = kspace.abs().max().item()
        sigma = ks_max / noise_snr
        kspace = kspace + sigma * torch.randn_like(kspace)

    n_x, n_y = params.get('n_x', 128), params.get('n_y', 128)
    img = mr0.reco_adjoint(signal, kspace, resolution=(n_x, n_y, 1), FOV=(fov, fov, 1.0))
    img = img.abs().squeeze()

    # Target + MAE (computed once from baseline, cached for all exps)
    global _baseline_target, _baseline_masks
    if exp_id == 1:
        _baseline_target, _baseline_masks = theoretical_target(phantom, TE, TR)
    target, masks = _baseline_target, _baseline_masks
    mae, per_tissue, img_scaled, global_scale = compute_mae(img, target, masks)

    # SNR, CNR, SAR
    bg = ~masks['all']
    noise = img[bg].std().item() if bg.sum() > 0 else 1.0
    snr = img[masks['all']].mean().item() / max(noise, 1e-8) if masks['all'].sum() > 0 else 0
    cnr = 0
    if masks['GM'].sum() > 0 and masks['WM'].sum() > 0 and bg.sum() > 0:
        cnr = (img[masks['GM']].mean() - img[masks['WM']].mean()).abs().item() / max(noise, 1e-8)

    sar_est = 0.001
    flips = params.get('rf_flip_angles', [180])
    sar_est = sum((f / 180) ** 2 for f in flips) * 2e-3 / TR

    acq_t = acq_time(params)

    # Aliasing
    aliasing = (img[bg].abs().mean() / max(img[masks['all']].abs().mean(), 1e-8)).item()

    metrics = {
        'status': 'ok', 'mae_total': mae, 'mae_per_tissue': per_tissue,
        'snr': snr, 'cnr_gm_wm': cnr, 'sar_estimate': sar_est,
        'acq_time_s': acq_t, 'aliasing_ratio': aliasing,
        'sim_time_s': time.time() - t0, 'timing_check': 'PASS' if ok else 'FAIL',
    }

    if not fast_mode:
        _save_image(img_scaled, target, masks, params, metrics, output_dir, exp_id, global_scale)

    del signal, kspace, img
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def _save_image(img, target, masks, params, metrics, output_dir, exp_id, global_scale=1.0):
    """Generate a 2x3 report panel: simulated + target + diff + tissue masks + per-tissue MAE + info."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    img_np = img.cpu().numpy()
    tgt_np = target.cpu().numpy()
    diff_np = np.abs(img_np - tgt_np)
    mask_all = masks['all'].cpu().numpy()

    vmin = min(img_np[mask_all].min(), tgt_np[mask_all].min())
    vmax = max(img_np[mask_all].max(), tgt_np[mask_all].max())

    # (0,0): Simulated (scaled)
    im0 = axes[0, 0].imshow(img_np, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    axes[0, 0].set_title('Simulated (scaled)')
    plt.colorbar(im0, ax=axes[0, 0])

    # (0,1): Theoretical target (same range, now comparable)
    im1 = axes[0, 1].imshow(tgt_np, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    te_ms = params.get('te', 0.08) * 1000
    tr_ms = params.get('tr', 3.0) * 1000
    axes[0, 1].set_title(f'Target (TE={te_ms:.0f}ms TR={tr_ms:.0f}ms)')
    plt.colorbar(im1, ax=axes[0, 1])

    # (0,2): |Difference|
    im2 = axes[0, 2].imshow(diff_np, cmap='hot', origin='lower')
    axes[0, 2].set_title(f'|Difference|  MAE={metrics["mae_total"]:.4f}')
    plt.colorbar(im2, ax=axes[0, 2])

    # (1,0): Tissue masks overlay
    mask_rgb = np.zeros((*img_np.shape, 3))
    overlays = {'CSF': [0, 0, 1], 'GM': [0, 1, 0], 'WM': [1, 0, 0]}
    for name, color in overlays.items():
        if name in masks:
            m = masks[name].cpu().numpy()
            for c in range(3):
                mask_rgb[..., c] += m * color[c] * 0.4
    axes[1, 0].imshow(img_np, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    axes[1, 0].imshow(mask_rgb, origin='lower', alpha=0.6)
    axes[1, 0].set_title('Tissue Masks (R=WM, G=GM, B=CSF)')

    # (1,1): Per-tissue MAE bar chart
    tissues = []; maes = []
    for name in ['CSF', 'GM', 'WM']:
        if name in metrics.get('mae_per_tissue', {}):
            tissues.append(name); maes.append(metrics['mae_per_tissue'][name])
    if tissues:
        axes[1, 1].bar(tissues, maes, color=['blue', 'green', 'red'])
        axes[1, 1].set_title('MAE per Tissue')
        axes[1, 1].set_ylabel('MAE')

    # (1,2): Info panel
    axes[1, 2].axis('off')
    flips = params.get('rf_flip_angles', params.get('flip_angle', 'N/A'))
    flip_str = str(flips[:4])[:-1] + '...]' if isinstance(flips, list) and len(flips) > 4 else str(flips)
    text = (f'Exp #{exp_id}\n{"="*25}\n'
            f'TE: {te_ms:.0f} ms\n'
            f'TR: {tr_ms:.0f} ms\n'
            f'Turbo: {params.get("n_echo","?")}\n'
            f'Flips: {flip_str}\n'
            f'Scale: {global_scale:.3f}\n'
            f'MAE total: {metrics["mae_total"]:.4f}\n'
            f'SNR: {metrics.get("snr",0):.1f}\n'
            f'CNR: {metrics.get("cnr_gm_wm",0):.1f}\n'
            f'SAR: {metrics.get("sar_estimate",0):.4f}\n'
            f'Acq: {metrics.get("acq_time_s",0):.0f} s')
    axes[1, 2].text(0.05, 0.95, text, transform=axes[1, 2].transAxes,
                    fontsize=9, fontfamily='monospace', verticalalignment='top')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'experiment_{exp_id:04d}.png'), dpi=100, bbox_inches='tight')
    plt.close(fig)
