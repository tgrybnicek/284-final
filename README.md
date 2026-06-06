# FHR extraction from Transabdominal PPG
## Implementation

The signal is processed one window at a time. Each window is 60 seconds long and
the window slides forward 30 seconds at a time. Before any of that, every channel
is bandpassed between 0.2 and 15 Hz with a zero phase Butterworth filter to strip
out baseline drift and high frequency junk.

For each window the code first checks quality. It measures spectral flatness across
the eight detectors, and if the window looks too much like noise it gets skipped so
a bad stretch never blotches the estimate.

If the window passes, three independent methods run on all eight detector signals:

1. ANC uses an RLS adaptive filter to cancel the maternal reference of the matching
   wavelength, then reads the dominant in band frequency of what is left.
2. EMD breaks the signal into oscillatory modes and takes the strongest peak inside
   the fetal heart rate band.
3. VMD does the same peak picking on a variational decomposition.

That produces up to 24 candidate estimates per window (8 detectors times 3
methods). They get combined by fuse, which takes a weighted median, throws out
anything more than three times the median absolute deviation away, and returns a
weighted mean of what survives. The weights blend three things: a spatial weight
per detector (D3 is trusted most), a weight per method, and each window's in band
SNR, so cleaner channels and stronger methods are more pronounced.

The fused value for each window forms a time series, which then gets smoothed
twice. An exponential moving average then applies a final smoothing.

The EMA has a tunable alpha factor. We sweep alpha over the two
training rounds (PPG1 and PPG2), score each value with a weighted, normalized MAE,
and keep whichever alpha scores best. The single alpha is then applied to every
dataset, including PPG3, which is held out and never used for tuning. Accuracy is
reported as MAE and RMSE in BPM against the reference signal.

## Figures

Four figures are produced, one per stage of the pipeline. The time series plots
use the PPG3 test set, where grey is the true reference FHR and the coloured line is
the estimate.

1. Stage 1 shows the raw ANC, EMD, and VMD estimates from a single detector against
   the truth, so we can see how scattered one channel and method is on its own.
2. Stage 2 shows the fusion across all 10 channels, with a line per method plus the
   combined all method line, showing the affects of fusion.
3. Stage 3 shows the smoothing chain, the fused track followed by the EMA result, with a small inset of the alpha sweep that picked the
   smoothing factor.
4. Stage 4 shows the final estimate against the truth for all three datasets, plus a
   bar panel of MAE by stage on the test set so the error visibly drops as each
   stage is added.
