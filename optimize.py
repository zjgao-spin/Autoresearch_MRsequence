"""
optimize.py — the editable file (equivalents: karpathy's train.py).
YOU (the agent) edit this file between runs.  Do NOT create new .py files.

Run:
    python optimize.py

The file is deliberately minimal so you can focus on parameter decisions.
"""
import sys; sys.path.insert(0, ".")
from autoresearch_mrsequence.evaluate import evaluate, score, acq_time
from autoresearch_mrsequence.sequences import SEQ_BUILDERS

N = 30     # total experiments (you may change this)
output_dir = "output"

# ============================================================================
# EDITABLE PARAMETERS — extract TE, TR, matrix from user instruction
# ============================================================================
best_params = {
    "fov": 0.20, "n_x": 128, "n_y": 128, "n_echo": 8,
    "rf_flip_angles": [180, 180, 180, 180, 180, 180, 180, 180],
    "slice_thickness": 5e-3, "te": 0.080, "tr": 3.0,
    "fsp_r": 1.0, "fsp_s": 0.5, "encoding": "linear",
}

# ============================================================================
# Baseline — DO NOT EDIT (exp_id=1 is always the baseline)
# ============================================================================
m1 = evaluate(best_params, output_dir, exp_id=1)
best_score = score(m1["mae_total"], m1["sar_estimate"], m1["acq_time_s"],
                   m1["mae_total"], m1["sar_estimate"], m1["acq_time_s"])
print(f"Baseline  MAE={m1['mae_total']:.4f}  Score={best_score:.4f}")

# ============================================================================
# EXPERIMENTS — edit the params for each experiment below
# ============================================================================
# You may vary: encoding, n_echo, rf_flip_angles, fsp_r, fsp_s
# When changing n_echo, you MUST also change rf_flip_angles to same length.

improvements = 0
for exp in range(2, N + 1):

    # ---- EDIT BELOW: start from best_params, then change what you want ----
    params = dict(best_params)

    # Add your experiments here.  Examples (commented out):
    #
    # if exp == 2:
    #     params["encoding"] = "..."          # linear or centric
    #     params["n_echo"] = ...              # 2, 4, 8, or 16
    #     params["rf_flip_angles"] = [...]    # length == n_echo
    #
    # if exp == 3:
    #     params["encoding"] = "..."
    #     params["fsp_r"] = ...
    #     params["fsp_s"] = ...
    #
    # ... add more experiments as needed ...
    #
    # ---- EDIT ABOVE ----

    m = evaluate(params, output_dir, exp_id=exp, fast_mode=True)
    s = score(m["mae_total"], m["sar_estimate"], m["acq_time_s"],
              m1["mae_total"], m1["sar_estimate"], m1["acq_time_s"])
    print(f"Exp {exp:3d}  MAE={m['mae_total']:.4f}  Score={s:.4f}")

    if 0 < s < best_score:
        best_score = s; best_params = dict(params)
        evaluate(best_params, output_dir, exp_id=exp, fast_mode=False)
        print(f"  -> KEEP (#{exp})")
        improvements += 1

# ============================================================================
# Save — DO NOT EDIT below unless adding optional reports
# ============================================================================
build_params = {k: v for k, v in best_params.items() if k != 'noise_snr'}
seq, ok, _, _ = SEQ_BUILDERS["tse"](**build_params)
seq.write(f"{output_dir}/best_sequence.seq")
print(f"\nDone.  {improvements} improvements in {N} experiments.")
print(f"Best Score={best_score:.4f}")
