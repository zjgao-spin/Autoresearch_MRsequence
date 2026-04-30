"""
2D Turbo Spin Echo sequence builder using PyPulseq.
Based on pypulseq write_tse.py reference implementation.
This file defines the sequence parameters that the agent modifies.
"""

import warnings
import numpy as np
import pypulseq as pp


def build_tse_sequence(
    fov: float = 220e-3,
    n_x: int = 64,
    n_y: int = 64,
    n_echo: int = 8,
    n_slices: int = 1,
    rf_flip_angles: list = None,
    slice_thickness: float = 5e-3,
    te: float = 80e-3,
    tr: float = 3000e-3,
    fsp_r: float = 1.0,
    fsp_s: float = 0.5,
    encoding: str = 'linear',
):
    """
    Build a 2D TSE sequence.

    Parameters (agent-editable):
    ----------
    fov : float
        Field of view in meters (square FOV).
    n_x : int
        Number of readout samples.
    n_y : int
        Number of phase encoding steps.
    n_echo : int
        Turbo factor (number of echoes per excitation).
    n_slices : int
        Number of slices.
    rf_flip_angles : list of float
        Refocusing flip angles in degrees (length = n_echo).
        If None, defaults to all 180°.
    slice_thickness : float
        Slice thickness in meters.
    te : float
        Echo time (effective TE, seconds).
    tr : float
        Repetition time (seconds).
    fsp_r : float
        Readout spoiler factor.
    fsp_s : float
        Slice spoiler factor.
    encoding : str
        'linear' (default) or 'centric' k-space view ordering.
        Centric puts ky=0 at first echo for pure T2 contrast (TE_eff=TE).

    Returns
    -------
    seq : pypulseq.Sequence
    """
    if rf_flip_angles is None:
        rf_flip_angles = [180.0] * n_echo
    else:
        n_echo = len(rf_flip_angles)

    dG = 250e-6  # gradient rise time

    system = pp.Opts(
        max_grad=32,
        grad_unit='mT/m',
        max_slew=130,
        slew_unit='T/m/s',
        rf_ringdown_time=100e-6,
        rf_dead_time=100e-6,
        adc_dead_time=10e-6,
    )

    seq = pp.Sequence(system)

    sampling_time = 6.4e-3
    readout_time = sampling_time + 2 * system.adc_dead_time
    t_ex = 2.5e-3
    t_exwd = t_ex + system.rf_ringdown_time + system.rf_dead_time
    t_ref = 2.0e-3
    t_refwd = t_ref + system.rf_ringdown_time + system.rf_dead_time
    t_sp = 0.5 * (te - readout_time - t_refwd)
    t_spex = 0.5 * (te - t_exwd - t_refwd)

    rf_ex_phase = np.pi / 2
    rf_ref_phase = 0

    # -- Excitation pulse --
    flip_ex = np.deg2rad(90)
    rf_ex, gz, _ = pp.make_sinc_pulse(
        flip_angle=flip_ex,
        system=system,
        duration=t_ex,
        slice_thickness=slice_thickness,
        apodization=0.5,
        time_bw_product=4,
        phase_offset=rf_ex_phase,
        return_gz=True,
        delay=system.rf_dead_time,
        use='excitation',
    )
    gs_ex = pp.make_trapezoid(
        channel='z', system=system,
        amplitude=gz.amplitude,
        flat_time=t_exwd,
        rise_time=dG,
    )

    # -- Refocusing pulse (template, angles vary per echo) --
    flip_ref = np.deg2rad(rf_flip_angles[0])
    rf_ref, gz_ref, _ = pp.make_sinc_pulse(
        flip_angle=flip_ref,
        system=system,
        duration=t_ref,
        slice_thickness=slice_thickness,
        apodization=0.5,
        time_bw_product=4,
        phase_offset=rf_ref_phase,
        use='refocusing',
        return_gz=True,
        delay=system.rf_dead_time,
    )
    gs_ref = pp.make_trapezoid(
        channel='z', system=system,
        amplitude=gs_ex.amplitude,
        flat_time=t_refwd,
        rise_time=dG,
    )

    ags_ex = gs_ex.area / 2
    gs_spr = pp.make_trapezoid(channel='z', system=system, area=ags_ex * (1 + fsp_s),
                               duration=t_sp, rise_time=dG)
    gs_spex = pp.make_trapezoid(channel='z', system=system, area=ags_ex * fsp_s,
                                duration=t_spex, rise_time=dG)

    delta_kx = 1 / fov
    delta_ky = 1 / fov
    k_width = n_x * delta_kx

    gr_acq = pp.make_trapezoid(channel='x', system=system, flat_area=k_width,
                               flat_time=readout_time, rise_time=dG)
    adc = pp.make_adc(num_samples=n_x, duration=sampling_time,
                      delay=system.adc_dead_time, system=system)
    gr_spr = pp.make_trapezoid(channel='x', system=system, area=gr_acq.area * fsp_r,
                               duration=t_sp, rise_time=dG)

    agr_spr = gr_spr.area
    agr_preph = gr_acq.area / 2 + agr_spr
    gr_preph = pp.make_trapezoid(channel='x', system=system, area=agr_preph,
                                 duration=t_spex, rise_time=dG)

    # -- Phase encoding order --
    n_ex = int(np.floor(n_y / n_echo))
    total_pe = n_echo * n_ex

    if encoding == 'centric':
        # Center-out reordering: build [0, +1, -1, +2, -2, ...] then
        # assign to excitation/echo grid using column-major (Fortran) order
        ky = np.arange(-n_y // 2, n_y // 2, dtype=float)
        center_out = []
        if 0 in ky:
            center_out.append(0.0)
        i = 1
        while len(center_out) < total_pe:
            if i in ky:
                center_out.append(float(i))
                if len(center_out) >= total_pe:
                    break
            if -i in ky:
                center_out.append(float(-i))
            i += 1
        pe_order = np.array(center_out).reshape((n_ex, n_echo), order='F').T
    else:
        # Linear Fortran-order (default)
        pe_steps = np.arange(1, total_pe + 1) - 0.5 * total_pe - 1
        if divmod(n_echo, 2)[1] == 0:
            pe_steps = np.roll(pe_steps, [0, int(-np.round(n_ex / 2))])
        pe_order = pe_steps.reshape((n_ex, n_echo), order='F').T

    phase_areas = pe_order * delta_ky

    # -- Extended trapezoid building blocks (slice/gradient merging) --
    gs1_times = np.array([0, gs_ex.rise_time])
    gs1_amp = np.array([0, gs_ex.amplitude])
    gs1 = pp.make_extended_trapezoid(channel='z', times=gs1_times, amplitudes=gs1_amp)

    gs2_times = np.array([0, gs_ex.flat_time])
    gs2_amp = np.array([gs_ex.amplitude, gs_ex.amplitude])
    gs2 = pp.make_extended_trapezoid(channel='z', times=gs2_times, amplitudes=gs2_amp)

    gs3_times = np.array([0, gs_spex.rise_time,
                          gs_spex.rise_time + gs_spex.flat_time,
                          gs_spex.rise_time + gs_spex.flat_time + gs_spex.fall_time])
    gs3_amp = np.array([gs_ex.amplitude, gs_spex.amplitude, gs_spex.amplitude, gs_ref.amplitude])
    gs3 = pp.make_extended_trapezoid(channel='z', times=gs3_times, amplitudes=gs3_amp)

    gs4_times = np.array([0, gs_ref.flat_time])
    gs4_amp = np.array([gs_ref.amplitude, gs_ref.amplitude])
    gs4 = pp.make_extended_trapezoid(channel='z', times=gs4_times, amplitudes=gs4_amp)

    gs5_times = np.array([0, gs_spr.rise_time,
                          gs_spr.rise_time + gs_spr.flat_time,
                          gs_spr.rise_time + gs_spr.flat_time + gs_spr.fall_time])
    gs5_amp = np.array([gs_ref.amplitude, gs_spr.amplitude, gs_spr.amplitude, 0])
    gs5 = pp.make_extended_trapezoid(channel='z', times=gs5_times, amplitudes=gs5_amp)

    gs7_times = np.array([0, gs_spr.rise_time,
                          gs_spr.rise_time + gs_spr.flat_time,
                          gs_spr.rise_time + gs_spr.flat_time + gs_spr.fall_time])
    gs7_amp = np.array([0, gs_spr.amplitude, gs_spr.amplitude, gs_ref.amplitude])
    gs7 = pp.make_extended_trapezoid(channel='z', times=gs7_times, amplitudes=gs7_amp)

    gr3 = gr_preph
    gr5_times = np.array([0, gr_spr.rise_time,
                          gr_spr.rise_time + gr_spr.flat_time,
                          gr_spr.rise_time + gr_spr.flat_time + gr_spr.fall_time])
    gr5_amp = np.array([0, gr_spr.amplitude, gr_spr.amplitude, gr_acq.amplitude])
    gr5 = pp.make_extended_trapezoid(channel='x', times=gr5_times, amplitudes=gr5_amp)

    gr6_times = np.array([0, readout_time])
    gr6_amp = np.array([gr_acq.amplitude, gr_acq.amplitude])
    gr6 = pp.make_extended_trapezoid(channel='x', times=gr6_times, amplitudes=gr6_amp)

    gr7_times = np.array([0, gr_spr.rise_time,
                          gr_spr.rise_time + gr_spr.flat_time,
                          gr_spr.rise_time + gr_spr.flat_time + gr_spr.fall_time])
    gr7_amp = np.array([gr_acq.amplitude, gr_spr.amplitude, gr_spr.amplitude, 0])
    gr7 = pp.make_extended_trapezoid(channel='x', times=gr7_times, amplitudes=gr7_amp)

    # -- Echo train timing --
    t_ex_dur = pp.calc_duration(gs1) + pp.calc_duration(gs2) + pp.calc_duration(gs3)
    t_ref_dur = pp.calc_duration(gs4) + pp.calc_duration(gs5) + pp.calc_duration(gs7) + readout_time
    t_end = pp.calc_duration(gs4) + pp.calc_duration(gs5)
    te_train = t_ex_dur + n_echo * t_ref_dur + t_end
    tr_delay = (tr - n_slices * te_train) / n_slices
    tr_delay = system.grad_raster_time * np.round(tr_delay / system.grad_raster_time)
    if tr_delay < 0:
        tr_delay = 1e-3
        warnings.warn(f'TR too short, adapted: {1000 * n_slices * (te_train + tr_delay):.0f} ms')
    else:
        print(f'TR delay: {1000 * tr_delay:.1f} ms')

    # -- Sequence assembly --
    for i_excitation in range(n_ex + 1):
        for i_slice in range(n_slices):
            rf_ex.freq_offset = gs_ex.amplitude * slice_thickness * (i_slice - (n_slices - 1) / 2)
            rf_ref.freq_offset = gs_ref.amplitude * slice_thickness * (i_slice - (n_slices - 1) / 2)
            rf_ex.phase_offset = rf_ex_phase - 2 * np.pi * rf_ex.freq_offset * pp.calc_rf_center(rf_ex)[0]
            rf_ref.phase_offset = rf_ref_phase - 2 * np.pi * rf_ref.freq_offset * pp.calc_rf_center(rf_ref)[0]

            seq.add_block(gs1)
            seq.add_block(rf_ex, gs2)
            seq.add_block(gs3, gr3)

            for i_echo in range(n_echo):
                if i_excitation > 0:
                    phase_area = phase_areas[i_echo, i_excitation - 1]
                else:
                    phase_area = 0.0

                gp_pre = pp.make_trapezoid(channel='y', system=system, area=phase_area,
                                           duration=t_sp, rise_time=dG)
                gp_rew = pp.make_trapezoid(channel='y', system=system, area=-phase_area,
                                           duration=t_sp, rise_time=dG)

                # Rebuild refocusing pulse for per-echo flip angle
                flip_ref_i = np.deg2rad(rf_flip_angles[i_echo])
                rf_ref_i, _, _ = pp.make_sinc_pulse(
                    flip_angle=flip_ref_i,
                    system=system,
                    duration=t_ref,
                    slice_thickness=slice_thickness,
                    apodization=0.5,
                    time_bw_product=4,
                    phase_offset=rf_ref.phase_offset,
                    use='refocusing',
                    return_gz=True,
                    delay=system.rf_dead_time,
                )

                seq.add_block(rf_ref_i, gs4)
                seq.add_block(gr5, gp_pre, gs5)
                if i_excitation > 0:
                    seq.add_block(gr6, adc)
                else:
                    seq.add_block(gr6)
                seq.add_block(gr7, gp_rew, gs7)

            seq.add_block(gs4)
            seq.add_block(gs5)
            seq.add_block(pp.make_delay(tr_delay))

    ok, error_report = seq.check_timing()
    if ok:
        print('Timing check passed')
    else:
        print('Timing check FAILED:')
        for e in error_report:
            print(f'  {e}')

    seq.set_definition(key='FOV', value=[fov, fov, slice_thickness * n_slices])
    seq.set_definition(key='Name', value='tse')

    return seq, ok, error_report, tr_delay


def get_default_params():
    return {
        'fov': 0.20,
        'n_x': 128,
        'n_y': 128,
        'n_echo': 8,
        'n_slices': 1,
        'rf_flip_angles': [180.0] * 8,
        'slice_thickness': 5e-3,
        'te': 80e-3,
        'tr': 3000e-3,
        'fsp_r': 1.0,
        'fsp_s': 0.5,
        'encoding': 'linear',
    }

TSE_DEFAULTS = get_default_params()
TSE_PARAMS = {
    'rf_flip_angles': {'type': 'list', 'range': [20, 180], 'desc': 'Refocusing flip angles (deg)',
                       'list_length_key': 'n_echo', 'perturb_mag': 15},
    'n_echo': {'type': 'int', 'valid': [2, 4, 8, 16], 'desc': 'Turbo factor (must divide n_y)'},
    'te': {'type': 'float', 'range': [30e-3, 150e-3], 'desc': 'Echo time (s)'},
    'tr': {'type': 'float', 'range': [1.5, 5.0], 'desc': 'Repetition time (s)'},
    'n_x': {'type': 'int', 'range': [64, 192], 'desc': 'Readout matrix'},
    'n_y': {'type': 'int', 'range': [64, 192], 'desc': 'Phase encode matrix'},
    'fov': {'type': 'float', 'range': [0.15, 0.30], 'desc': 'Field of view (m)'},
    'slice_thickness': {'type': 'float', 'range': [3e-3, 8e-3], 'desc': 'Slice thickness (m)'},
    'fsp_r': {'type': 'float', 'range': [0.3, 2.5], 'desc': 'Readout crusher factor'},
    'fsp_s': {'type': 'float', 'range': [0.1, 2.0], 'desc': 'Slice crusher factor'},
    'encoding': {'type': 'choice', 'choices': ['linear', 'centric'], 'desc': 'k-space view order'},
}
