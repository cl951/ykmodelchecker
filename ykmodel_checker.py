import pickle
import numpy as np
import flopy

def save_reference_model(mf, filepath):
    """
    Extracts key numerical data from a FloPy MODFLOW model and saves it
    as a plain dictionary. This is more robust than pickling the full mf
    object, which carries file handles and solver state.

    Arguments:
    - mf       : FloPy MODFLOW model object
    - filepath : path to the output .pkl file
    """
    ref = {}

    # ── Discretisation ──────────────────────────────────────────────────
    dis = mf.get_package('DIS')
    ref['dis'] = {
        'nlay'   : int(dis.nlay),
        'nrow'   : int(dis.nrow),
        'ncol'   : int(dis.ncol),
        'delr'   : np.array(dis.delr.array),
        'delc'   : np.array(dis.delc.array),
        'top'    : np.array(dis.top.array),
        'botm'   : np.array(dis.botm.array),
        'nper'   : int(dis.nper),
        'perlen' : np.array(dis.perlen.array),
        'nstp'   : np.array(dis.nstp.array),
        'steady' : np.array(dis.steady.array),
    }

    # ── Basic (ibound, starting heads) ──────────────────────────────────
    bas = mf.get_package('BAS6')
    ref['bas'] = {
        'ibound' : np.array(bas.ibound.array),
        'strt'   : np.array(bas.strt.array),
    }

    # ── Layer-Property Flow ──────────────────────────────────────────────
    lpf = mf.get_package('LPF')
    ref['lpf'] = {
        'hk'     : np.array(lpf.hk.array),
        'vka'    : np.array(lpf.vka.array),
        'ss'     : np.array(lpf.ss.array),
        'sy'     : np.array(lpf.sy.array),
        'layvka' : np.array(lpf.layvka),
    }

    # ── Recharge ─────────────────────────────────────────────────────────
    rch = mf.get_package('RCH')
    ref['rch'] = {
        'rech' : np.array(rch.rech.array),
    }

    # ── Rivers ───────────────────────────────────────────────────────────
    riv = mf.get_package('RIV')
    if riv is not None:
        # Flatten all stress-period data into one sorted array for comparison
        all_riv = []
        for sp_data in riv.stress_period_data.data.values():
            for rec in sp_data:
                all_riv.append([rec['k'], rec['i'], rec['j'],
                                 rec['stage'], rec['cond'], rec['rbot']])
        ref['riv'] = np.array(sorted(all_riv))
    else:
        ref['riv'] = None

    # ── Wells ─────────────────────────────────────────────────────────────
    wel = mf.get_package('WEL')
    if wel is not None:
        all_wel = []
        for sp_data in wel.stress_period_data.data.values():
            for rec in sp_data:
                all_wel.append([rec['k'], rec['i'], rec['j'], rec['flux']])
        ref['wel'] = np.array(sorted(all_wel))
    else:
        ref['wel'] = None

    with open(filepath, 'wb') as f:
        pickle.dump(ref, f)

    print(f"Reference model saved to: {filepath}")

def compare_to_reference(mf, filepath, rtol=1e-5, atol=1e-8):
    """
    Loads the reference model dictionary and compares it to the student's
    mf object, printing a clear pass/fail report for every parameter.

    Arguments:
    - mf       : Student's FloPy MODFLOW model object
    - filepath : Path to the reference .pkl file saved by the instructor
    - rtol     : Relative tolerance for floating-point comparisons
    - atol     : Absolute tolerance for floating-point comparisons
    """

    with open(filepath, 'rb') as f:
        ref = pickle.load(f)

    issues   = []   # collects descriptions of problems found
    n_checks = 0    # total number of checks performed

    # ── Comparison helpers ────────────────────────────────────────────────

    def _check_scalar(pkg, name, student_val, ref_val):
        nonlocal n_checks
        n_checks += 1
        if student_val != ref_val:
            issues.append(
                f"[{pkg}] '{name}': yours = {student_val}, reference = {ref_val}"
            )

    def _check_array(pkg, name, student_arr, ref_arr):
        nonlocal n_checks
        n_checks += 1
        #student_arr = np.array(student_arr, dtype=float)
        #ref_arr     = np.array(ref_arr,     dtype=float)

        if student_arr.shape != ref_arr.shape:
            issues.append(
                f"[{pkg}] '{name}' shape mismatch: "
                f"yours = {student_arr.shape}, reference = {ref_arr.shape}"
            )
            return

        # Use nan-safe comparison
        diff_mask   = ~np.isclose(student_arr, ref_arr,
                                   rtol=rtol, atol=atol,
                                   equal_nan=True)
        n_diff      = int(np.sum(diff_mask))
        n_total     = int(diff_mask.size)

        if n_diff > 0:
            max_diff    = float(np.nanmax(np.abs(student_arr - ref_arr)))
            mean_diff   = float(np.nanmean(np.abs(student_arr[diff_mask]
                                                   - ref_arr[diff_mask])))
            issues.append(
                f"[{pkg}] '{name}': {n_diff}/{n_total} cells differ  "
                f"(max |Δ| = {max_diff:.4g}, mean |Δ| = {mean_diff:.4g})"
            )

    # ── DIS ───────────────────────────────────────────────────────────────
    dis = mf.get_package('DIS')
    r   = ref['dis']

    _check_scalar('DIS', 'nlay',   dis.nlay,              r['nlay'])
    _check_scalar('DIS', 'nrow',   dis.nrow,              r['nrow'])
    _check_scalar('DIS', 'ncol',   dis.ncol,              r['ncol'])
    _check_scalar('DIS', 'nper',   dis.nper,              r['nper'])
    _check_array ('DIS', 'delr',   dis.delr.array,        r['delr'])
    _check_array ('DIS', 'delc',   dis.delc.array,        r['delc'])
    _check_array ('DIS', 'top',    dis.top.array,         r['top'])
    _check_array ('DIS', 'botm',   dis.botm.array,        r['botm'])
    _check_array ('DIS', 'perlen', dis.perlen.array,      r['perlen'])
    _check_array ('DIS', 'nstp',   dis.nstp.array,        r['nstp'])
    _check_array ('DIS', 'steady', dis.steady.array,      r['steady'])

    # ── BAS ───────────────────────────────────────────────────────────────
    bas = mf.get_package('BAS6')
    r   = ref['bas']

    _check_array('BAS6', 'ibound', bas.ibound.array, r['ibound'])
    _check_array('BAS6', 'strt',   bas.strt.array,   r['strt'])

    # ── LPF ───────────────────────────────────────────────────────────────
    lpf = mf.get_package('LPF')
    r   = ref['lpf']

    _check_array('LPF', 'hk',     lpf.hk.array,    r['hk'] )
    _check_array('LPF', 'vka',    lpf.vka.array,   r['vka'] )
    _check_array('LPF', 'ss',     lpf.ss.array,    r['ss'] )
    _check_array('LPF', 'sy',     lpf.sy.array,    r['sy'] )
    _check_array('LPF', 'layvka',  lpf.layvka.array, r['layvka'] )

    # ── RCH ───────────────────────────────────────────────────────────────
    rch = mf.get_package('RCH')
    if rch is not None:
        _check_array('RCH', 'rech', rch.rech.array, ref['rch']['rech'] )
    else:
        if ref['rch'] is not None and 'rech' in ref['rch']:
            issues.append("[RCH] Package missing from your model.")

    # ── RIV ───────────────────────────────────────────────────────────────
    riv     = mf.get_package('RIV')
    ref_riv = ref['riv']
    n_checks += 1

    if riv is None and ref_riv is not None:
        issues.append("[RIV] Package missing from your model.")
    elif riv is not None and ref_riv is None:
        issues.append("[RIV] Unexpected river package found in your model.")
    elif riv is not None and ref_riv is not None:
        all_riv = []
        for sp_data in riv.stress_period_data.data.values():
            for rec in sp_data:
                all_riv.append([rec['k'], rec['i'], rec['j'],
                                 rec['stage'], rec['cond'], rec['rbot']])
        student_riv = np.array(sorted(all_riv))
        _check_array('RIV', 'stress_period_data (all cells)', student_riv, ref_riv)

    # ── WEL ───────────────────────────────────────────────────────────────
    wel     = mf.get_package('WEL')
    ref_wel = ref['wel']
    n_checks += 1

    if wel is None and ref_wel is not None:
        issues.append("[WEL] Package missing from your model.")
    elif wel is not None and ref_wel is None:
        issues.append("[WEL] Unexpected well package found in your model.")
    elif wel is not None and ref_wel is not None:
        all_wel = []
        for sp_data in wel.stress_period_data.data.values():
            for rec in sp_data:
                all_wel.append([rec['k'], rec['i'], rec['j'], rec['flux']])
        student_wel = np.array(sorted(all_wel))
        _check_array('WEL', 'stress_period_data (all wells)', student_wel, ref_wel)

    # ── Report ────────────────────────────────────────────────────────────
    print("=" * 60)
    print("  MODEL COMPARISON REPORT")
    print("=" * 60)

    if not issues:
        print(f"  ✅  All {n_checks} checks passed \u2014 your model matches the reference.")
    else:
        print(f"  \u26A0\uFE0F  {len(issues)} issue(s) found in {n_checks} checks:\n")
        for i, msg in enumerate(issues, 1):
            print(f"  {i:>2}. {msg}")
        print()
        print("  Tip: re-read the relevant code block carefully and check")
        print("  the parameter values against those in the tutorial notes.")

    print("=" * 60)
    return issues   # return list so it can be inspected programmatically

