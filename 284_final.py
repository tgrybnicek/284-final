import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
from PyEMD import EMD
from vmdpy import VMD
from sklearn.metrics import mean_absolute_error, mean_squared_error

FS = 80.0
FHR_MIN, FHR_MAX = 110, 270          # estimated fetal HR range, bpm
WIN_SEC, STEP_SEC = 60, 30
SQI_THRESH = 0.25
WINDOW_SAMPLES = int(WIN_SEC * FS)
STEP_SAMPLES = int(STEP_SEC * FS)

# detectors D2-D5, D3 sits closest to fetus, most weight
# both wavelengths use same weights, 8 signals total
SPATIAL_WEIGHTS = np.tile([1.0, 3.0, 2.0, 2.0], 2)   # WL1 D2-D5, then WL2 D2-D5
METHOD_WEIGHTS = {'ANC': 1.0, 'EMD': 1.5, 'VMD': 2.0}
TRAIN_WEIGHTS = {'PPG1': 0.7, 'PPG2': 0.3}
DAMPING_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
TRAIN_NAMES = ['PPG1', 'PPG2']
ALL_NAMES = ['PPG1', 'PPG2', 'PPG3']

HZ_LO, HZ_HI = FHR_MIN / 60.0, FHR_MAX / 60.0


def bandpass(x, lo=0.2, hi=15.0, order=4):
    nyq = 0.5 * FS
    b, a = signal.butter(order, [lo / nyq, hi / nyq], btype='bandpass')
    return signal.filtfilt(b, a, x)


def spectral_flatness(win):
    # 1 = noise like, 0 means tonal
    _, psd = signal.welch(win, FS, nperseg=min(len(win), 512))
    psd += 1e-12
    return float(np.exp(np.mean(np.log(psd))) / np.mean(psd))


def band_snr(win):
    freqs, psd = signal.welch(win, FS, nperseg=min(len(win), 512))
    band = (freqs >= HZ_LO) & (freqs <= HZ_HI)
    return float(psd[band].sum() / (psd.sum() + 1e-12))


def peak_bpm(freqs, psd):
    band = np.where((freqs >= HZ_LO) & (freqs <= HZ_HI))[0]
    if len(band) == 0:
        return np.nan
    return float(freqs[band][np.argmax(psd[band])] * 60.0)


def fuse(values, weights):
    # weighted median, reject anything > 3*MAD away, weighted mean of the rest
    v = np.array(values, dtype=float)
    w = np.array(weights, dtype=float)
    ok = ~np.isnan(v) & (w > 0)
    if not np.any(ok):
        return np.nan
    v, w = v[ok], w[ok] / w[ok].sum()
    order = np.argsort(v)
    med = v[order][np.searchsorted(np.cumsum(w[order]), 0.5)]
    mad = np.median(np.abs(v - med))
    thr = 3 * mad if mad > 0 else 15.0
    keep = np.abs(v - med) <= thr
    if not np.any(keep):
        keep = np.ones(len(v), dtype=bool)
    return float(np.average(v[keep], weights=w[keep]))


def ema(arr, alpha):
    out = np.full_like(arr, np.nan)
    prev = None
    for i, v in enumerate(arr):
        if np.isnan(v):
            out[i] = prev if prev is not None else np.nan
        elif prev is None:
            out[i] = prev = v
        else:
            out[i] = prev = alpha * v + (1 - alpha) * prev
    return out


def metrics(est, ref):
    est, ref = np.array(est), np.array(ref)
    m = ~np.isnan(est) & ~np.isnan(ref)
    if m.sum() == 0:
        return np.nan, np.nan
    return mean_absolute_error(ref[m], est[m]), np.sqrt(mean_squared_error(ref[m], est[m]))


def rls(ref, mixed, taps=100, lam=0.999):
    # RLS adaptive filter
    w = np.zeros(taps)
    P = np.eye(taps)
    out = np.zeros(len(mixed))
    pad = np.pad(ref, (taps - 1, 0))
    for i in range(len(mixed)):
        x = pad[i:i + taps][::-1]
        e = mixed[i] - w @ x
        Px = P @ x
        k = Px / (lam + x @ Px)
        w = w + k * e
        P = (P - np.outer(k, x @ P)) / lam
        out[i] = e
    return out


def best_band_peak(components):
    # pick the strongest peak inside FHR band 
    best, best_p = np.nan, 0.0
    for c in components:
        freqs, psd = signal.welch(c, FS, nperseg=len(c))
        band = np.where((freqs >= HZ_LO) & (freqs <= HZ_HI))[0]
        if len(band) and psd[band].max() > best_p:
            best_p = psd[band].max()
            best = freqs[band][np.argmax(psd[band])] * 60.0
    return best


def anc_bpm(mat_win, det_win):
    freqs, psd = signal.welch(rls(mat_win, det_win), FS, nperseg=WINDOW_SAMPLES)
    return peak_bpm(freqs, psd)


def emd_bpm(det_win):
    return best_band_peak(EMD().emd(det_win, max_imf=6))


def vmd_bpm(det_win):
    try:
        modes, _, _ = VMD(det_win, 2000, 0., 4, 0, 1, 1e-6)
    except Exception:
        return np.nan
    return best_band_peak(modes)


class KalmanTracker:
    # constant velocity model on [bpm, bpm rate]
    def __init__(self, dt=STEP_SEC, q=5.0, r=10.0):
        self.F = np.array([[1, dt], [0, 1]], float)
        self.H = np.array([[1, 0]], float)
        self.Q = np.eye(2) * q
        self.R = np.array([[r]])
        self.x = None
        self.P = np.eye(2) * 500.0

    def update(self, bpm):
        if self.x is None:
            self.x = np.array([bpm, 0.0])
            return bpm
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        y = np.array([[bpm]]) - self.H @ self.x.reshape(-1, 1)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y).flatten()
        self.P = (np.eye(2) - K @ self.H) @ self.P
        return float(self.x[0])

    def predict(self):
        if self.x is None:
            return np.nan
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return float(self.x[0])


def kalman_track(series):
    kf = KalmanTracker()
    return np.array([kf.update(v) if not np.isnan(v) else kf.predict() for v in series])


def load(name):
    ppg = pd.read_csv(f'{name}.csv')
    ref = pd.read_csv(f'FHR{name[-1]}.csv', header=None).values.flatten()
    # ch1 = maternal reference, both wavelengths
    mats = [bandpass(ppg[f'ch1volts{wl}'].values) for wl in ('WL1', 'WL2')]
    dets, refs = [], []
    for wi, wl in enumerate(('WL1', 'WL2')):
        for c in (2, 3, 4, 5):
            dets.append(bandpass(ppg[f'ch{c}volts{wl}'].values))
            refs.append(mats[wi])                                   # each detector keeps wavelength's reference
    return {'ref': ref, 'mats': refs, 'dets': dets}


def starts(n):
    return range(0, n - WINDOW_SAMPLES, STEP_SAMPLES)


def center_min(start):
    return ((start + WINDOW_SAMPLES / 2) / FS) / 60.0


DET3 = 1   # D3 at WL1 - single best detector, kept for the stage-1 baseline plot


def process(ds):
    # run all three methods 
    mats, dets, ref = ds['mats'], ds['dets'], ds['ref']
    ref_t = np.arange(len(ref)) / 60.0
    t, gt, gated = [], [], []
    a3, e3, v3 = [], [], []                 # raw single-detector estimates (stage 1)
    fa, fe, fv, fall = [], [], [], []       # fused estimates (stage 2)
    for s in starts(len(dets[0])):
        dw = [d[s:s + WINDOW_SAMPLES] for d in dets]
        mw = [m[s:s + WINDOW_SAMPLES] for m in mats]
        noisy = np.mean([spectral_flatness(w) for w in dw]) > SQI_THRESH
        t.append(center_min(s))
        gt.append(float(np.interp(t[-1], ref_t, ref)))
        gated.append(noisy)
        if noisy:
            for lst in (a3, e3, v3, fa, fe, fv, fall):
                lst.append(np.nan)
            continue
        snr = np.array([band_snr(w) for w in dw])
        anc = [anc_bpm(mw[i], w) for i, w in enumerate(dw)]
        emd = [emd_bpm(w) for w in dw]
        vmd = [vmd_bpm(w) for w in dw]
        a3.append(anc[DET3]); e3.append(emd[DET3]); v3.append(vmd[DET3])
        fa.append(fuse(anc, SPATIAL_WEIGHTS * snr))
        fe.append(fuse(emd, SPATIAL_WEIGHTS * snr))
        fv.append(fuse(vmd, SPATIAL_WEIGHTS * snr))
        vals, wts = [], []
        for mname, est in (('ANC', anc), ('EMD', emd), ('VMD', vmd)):
            for di, e in enumerate(est):
                vals.append(e)
                wts.append(METHOD_WEIGHTS[mname] * SPATIAL_WEIGHTS[di] * snr[di])
        fall.append(fuse(vals, wts))
    return {k: np.array(v) for k, v in
            dict(t=t, gt=gt, gated=gated,
                 anc_d3=a3, emd_d3=e3, vmd_d3=v3,
                 fused_anc=fa, fused_emd=fe, fused_vmd=fv, fused_all=fall).items()}


def main():
    print('loading data...')
    data = {n: load(n) for n in ALL_NAMES}

    # full 4 detector fusion, once per dataset
    print('processing...')
    proc = {}
    for n in ALL_NAMES:
        print('  ' + n)
        proc[n] = process(data[n])

    # Kalman smooth fused track, sweep EMA factor on training sets
    print('damping sweep...')
    kal_train = {n: kalman_track(proc[n]['fused_all']) for n in TRAIN_NAMES}
    sweep = {}
    for a in DAMPING_VALUES:
        score = total = 0.0
        for name in TRAIN_NAMES:
            gt = proc[name]['gt']
            d = ema(kal_train[name], a)
            m = ~np.isnan(d) & ~np.isnan(gt)
            if not m.sum():
                continue
            mae, _ = metrics(d, gt)
            score += TRAIN_WEIGHTS[name] * (mae / np.mean(gt[m]))   # normalize so PPG1/PPG2 are comparable
            total += TRAIN_WEIGHTS[name]
        sweep[a] = score / total if total else np.inf
    best_alpha = min(sweep, key=sweep.get)
    print(f'  best alpha = {best_alpha}')

    # apply alpha 
    final = {}
    for name in ALL_NAMES:
        kal = kalman_track(proc[name]['fused_all'])
        final[name] = {'kalman': kal, 'damped': ema(kal, best_alpha)}

    
    demo = 'PPG3'
    p, f = proc[demo], final[demo]
    t = p['t']
    ref = data[demo]['ref']
    ref_t = np.arange(len(ref)) / 60.0

    def add_truth(ax):
        ax.plot(ref_t, ref, color='0.6', lw=1, label='ground truth')

    # Stage 1: raw estimate on a single detector (D3)
    fig, ax = plt.subplots(figsize=(11, 4))
    add_truth(ax)
    for key, lab, col in (('anc_d3', 'ANC', 'tab:orange'),
                          ('emd_d3', 'EMD', 'tab:green'),
                          ('vmd_d3', 'VMD', 'tab:red')):
        mae, _ = metrics(p[key], p['gt'])
        ax.plot(t, p[key], '.', ms=4, color=col, alpha=0.7, label=f'{lab} (MAE={mae:.1f})')
    ax.set_title(f'Stage 1 - single detector (D3), raw method estimates [{demo}]')
    ax.set_xlabel('time (min)'); ax.set_ylabel('FHR (bpm)')
    ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.3)
    fig.tight_layout()

    # Stage 2:  all 10 channels (per method,  all methods)
    fig, ax = plt.subplots(figsize=(11, 4))
    add_truth(ax)
    for key, lab, col in (('fused_anc', 'ANC fused', 'tab:orange'),
                          ('fused_emd', 'EMD fused', 'tab:green'),
                          ('fused_vmd', 'VMD fused', 'tab:red'),
                          ('fused_all', 'all-method fused', 'tab:blue')):
        mae, _ = metrics(p[key], p['gt'])
        lw = 2.0 if key == 'fused_all' else 1.0
        ax.plot(t, p[key], color=col, lw=lw, label=f'{lab} (MAE={mae:.1f})')
    ax.set_title(f'Stage 2 - sensor fusion across 10 channels [{demo}]')
    ax.set_xlabel('time (min)'); ax.set_ylabel('FHR (bpm)')
    ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.3)
    fig.tight_layout()

    # Stage 3: temporal smoothing and kalman, EMA
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5),
                                   gridspec_kw={'width_ratios': [2, 1]})
    add_truth(ax1)
    ax1.plot(t, p['fused_all'], color='0.8', lw=1, label='fused (pre-smoothing)')
    ax1.plot(t, f['kalman'], 'm-', lw=1.2, label='+ Kalman')
    mae, _ = metrics(f['damped'], p['gt'])
    ax1.plot(t, f['damped'], 'b-', lw=2, label=f'+ EMA a={best_alpha} (MAE={mae:.1f})')
    ax1.set_title(f'Stage 3 - temporal smoothing [{demo}]')
    ax1.set_xlabel('time (min)'); ax1.set_ylabel('FHR (bpm)')
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)
    alphas = list(sweep)
    ax2.plot(alphas, [sweep[a] for a in alphas], 'o-')
    ax2.plot(best_alpha, sweep[best_alpha], 'r*', ms=15, label=f'best = {best_alpha}')
    ax2.set_title('EMA sweep (training sets)')
    ax2.set_xlabel('alpha'); ax2.set_ylabel('weighted norm. MAE')
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
    fig.tight_layout()

    # Stage 4: final result on every dataset + error summary
    fig, axes = plt.subplots(len(ALL_NAMES) + 1, 1, figsize=(11, 11))
    for ax, name in zip(axes, ALL_NAMES):
        pp, ff = proc[name], final[name]
        split = 'TEST' if name == 'PPG3' else 'train'
        mae, rmse = metrics(ff['damped'], pp['gt'])
        rr = data[name]['ref']
        ax.plot(np.arange(len(rr)) / 60.0, rr, color='0.6', lw=1, label='ground truth')
        ax.plot(pp['t'], ff['damped'], 'b-', lw=1.5,
                label=f'final estimate (MAE={mae:.1f}, RMSE={rmse:.1f})')
        ax.set_title(f'{name} ({split})')
        ax.set_ylabel('FHR (bpm)')
        ax.legend(loc='upper right', fontsize=8); ax.grid(True, alpha=0.3)
    # bottom panel: MAE on test set (PPG3)
    stage_series = [('Stage 1\n(D3 ANC)', p['anc_d3']),
                    ('Stage 2\n(fused all)', p['fused_all']),
                    ('Stage 3\n(Kalman)', f['kalman']),
                    ('Stage 4\n(+EMA)', f['damped'])]
    stage_maes = [metrics(s, p['gt'])[0] for _, s in stage_series]
    axes[-1].bar([lbl for lbl, _ in stage_series], stage_maes, color='tab:blue')
    for i, v in enumerate(stage_maes):
        axes[-1].text(i, v, f'{v:.1f}', ha='center', va='bottom', fontsize=8)
    axes[-1].set_ylabel('MAE (bpm)')
    axes[-1].set_title(f'Stage 4 - MAE by pipeline stage ({demo} test set)')
    axes[-1].grid(True, axis='y', alpha=0.3)
    axes[-2].set_xlabel('time (min)')
    fig.suptitle(f'Stage 4 - final FHR estimate (alpha={best_alpha})')
    fig.tight_layout()

    # print metrics to console
    print('\nfinal metrics (MAE / RMSE in bpm):')
    for name in ALL_NAMES:
        p = proc[name]
        split = 'test' if name == 'PPG3' else 'train'
        for label, est in (('ANC fused (8ch)', p['fused_anc']),
                           ('EMD fused (8ch)', p['fused_emd']),
                           ('VMD fused (8ch)', p['fused_vmd']),
                           ('All methods fused', p['fused_all']),
                           ('Kalman (no EMA)', final[name]['kalman']),
                           (f'Kalman + EMA alpha={best_alpha}', final[name]['damped'])):
            mae, rmse = metrics(est, p['gt'])
            print(f'  {name:<5} {split:<5} {label:<28} {mae:6.2f}  {rmse:6.2f}')

    plt.show()


if __name__ == '__main__':
    main()