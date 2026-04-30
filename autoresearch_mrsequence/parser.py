"""Natural language instruction parser for MRI sequence generation."""

import re


def parse_instruction(text):
    """Parse instruction into sequence type, constraints, and objectives."""
    text = text.lower()
    result = {'constraints': {}, 'objectives': {}}

    # Sequence type
    if any(w in text for w in ['tse', 'turbo spin echo', 'fast spin echo', 'fse']):
        result['seq_type'] = 'tse'
    else:
        result['seq_type'] = 'tse'  # default

    # Matrix: "128x128 matrix" or "128x128" or "192 matrix"
    m = re.search(r'(\d+)\s*[x×]\s*(\d+)\s*ma?t?r?i?x?', text)
    if not m:
        m = re.search(r'(\d+)\s*[x×]\s*(\d+)', text)
    if m:
        result['constraints']['n_x'] = int(m.group(1))
        result['constraints']['n_y'] = int(m.group(2))
    else:
        m = re.search(r'ma?t?r?i?x?\s*[=: ]\s*(\d+)', text)
        if m:
            result['constraints']['n_x'] = int(m.group(1))
            result['constraints']['n_y'] = int(m.group(1))

    # FOV: "FOV=(0.2 0.2 0.005)m" or "FOV 0.22m"
    m = re.search(r'fov\s*[=:]\s*\(([\d.]+)\s*,?\s*([\d.]+)\s*,?\s*([\d.]*)\)', text)
    if m:
        result['constraints']['fov'] = float(m.group(1))
        if m.group(3):
            result['constraints']['slice_thickness'] = float(m.group(3))
    else:
        m = re.search(r'fov\s*[=: ]\s*([\d.]+)\s*(mm|cm|m)', text)
        if m:
            val = float(m.group(1))
            unit = m.group(2)
            if unit == 'mm':
                val *= 1e-3
            elif unit == 'cm':
                val *= 1e-2
            result['constraints']['fov'] = val

    # TE
    m = re.search(r'te\s*[=: ]\s*(\d+)\s*ms', text)
    if m:
        result['constraints']['te'] = float(m.group(1)) * 1e-3

    # TR
    m = re.search(r'tr\s*[=: ]\s*(\d+)\s*ms', text)
    if m:
        result['constraints']['tr'] = float(m.group(1)) * 1e-3

    # Time constraint: "within X min/sec"
    m = re.search(r'within\s+(\d+)\s*(min|sec|s|m)', text)
    if m:
        value = int(m.group(1))
        result['constraints']['max_acq_time'] = value * 60 if m.group(2) in ('min', 'm') else value

    # Turbo factor (TSE specific)
    m = re.search(r'turbo\s*[=: ]\s*(\d+)', text)
    if m:
        result['constraints']['n_echo'] = int(m.group(1))

    # Encoding
    if 'centric' in text:
        result['constraints']['encoding'] = 'centric'
    elif 'linear' in text:
        result['constraints']['encoding'] = 'linear'

    # Slice thickness
    m = re.search(r'slice\s*(?:thickness)?\s*[=: ]\s*(\d+)\s*mm', text)
    if not m:
        m = re.search(r'st\s*[=: ]\s*(\d+)\s*mm', text)
    if m:
        result['constraints']['slice_thickness'] = float(m.group(1)) * 1e-3

    # Contrast
    if 't1' in text and 't2' not in text:
        result['objectives']['contrast'] = 't1'
    elif 't2' in text:
        result['objectives']['contrast'] = 't2'
    elif 'pd' in text or 'proton density' in text:
        result['objectives']['contrast'] = 'pd'

    return result


def apply_constraints(params, constraints):
    """Merge parsed constraints into parameter dict."""
    p = dict(params)
    for key in constraints:
        if key in p:
            p[key] = constraints[key]
    if 'n_x' in constraints and 'n_y' not in constraints:
        p['n_y'] = p['n_x']
    if 'n_y' in constraints and 'n_x' not in constraints:
        p['n_x'] = p['n_y']
    return p


def get_defaults_for(seq_type):
    """Return default params and editable parameter metadata for sequence type."""
    from .sequences import SEQ_DEFAULTS, SEQ_PARAMS
    return SEQ_DEFAULTS.get(seq_type, SEQ_DEFAULTS['tse']), SEQ_PARAMS.get(seq_type, SEQ_PARAMS['tse'])
