import numpy as np
import matplotlib.pyplot as plt


class PeakDetect:

    def __init__(self, moving_avg_window=30, delta_amplitude_min=2, delta_amplitude_max=20, grouping_min_dist=100):
        self.moving_avg_window = moving_avg_window
        self.delta_amplitude_min = delta_amplitude_min
        self.delta_amplitude_max = delta_amplitude_max
        self.grouping_min_dist = grouping_min_dist

    def detect_peaks(self, signal):
        mov_avg = self.moving_average(signal) + 20
        mov_avg_delta = self.moving_average_delta(signal, mov_avg)
        delta_flat = self.flatten_moving_average_delta(mov_avg_delta)
        peaks = self.identify_peaks_from_moving_delta(delta_flat)
        peak_groups = self.peak_grouping(peaks)
        peaks = self.consolidate_peak_groups(signal, peak_groups)
        peaks_no_noise = self.remove_noise(peaks, signal)
        return peaks_no_noise

    def plot_peaks(self, peaks, signal):
        fig, ax = plt.subplots(figsize=(14, 4))
        y_peaks = [signal[x] for x in peaks]
        ax.plot(signal, 'g')
        ax.plot(peaks, y_peaks, 'ro')
        plt.show()
        return fig, ax

    def __call__(self, signal):
        return self.detect_peaks(signal)

    def moving_average(self, signal):
        window = self.moving_avg_window
        a = np.pad(signal, [1, window - 2], mode="mean")
        ret = np.cumsum(a, dtype=float)
        ret[window:] = ret[window:] - ret[:-window]
        return ret[window - 1:] / window

    def moving_average_delta(self, signal, moving_average):
        delta = []
        for a, s in zip(moving_average, signal):
            d = s - a
            delta.append(d)
        return delta

    def flatten_moving_average_delta(self, moving_average_delta):
        min_threshold = self.delta_amplitude_min
        max_threshold = self.delta_amplitude_max
        result = []
        for a in moving_average_delta:
            if a < min_threshold:
                result.append(0)
            elif a >= min_threshold and a <= max_threshold:
                result.append(1)
            else:
                result.append(0)
        return result

    # Peak Detection
    def identify_peaks_from_moving_delta(self, moving_avg_delta):
        peaks = [i for i, n in enumerate(moving_avg_delta) if n > 0]
        return peaks

    def peak_grouping(self, peaks):
        min_dist = self.grouping_min_dist
        peaks = list(set(peaks))
        peaks.sort()
        sections = []
        section = []
        for i in range(len(peaks) - 1):
            section.append(peaks[i])
            peak_delta = peaks[i + 1] - peaks[i]
            if peak_delta > min_dist:
                sections.append(section)
                section = []
        sections.append(section)
        return sections

    def consolidate_peak_groups(self, signal, groups):
        peak_indexes = []
        for group in groups:
            if len(group) > 1:
                group.sort()
                slice = signal[group[0]:group[-1]]
                slice_peak = np.argmax(slice)
                peak_index = slice_peak + group[0]
                peak_indexes.append(peak_index)
        return peak_indexes

    # Noise and Outlier Removal
    def remove_noise(self, peaks, signal):
        new_peaks = []
        for p in peaks:
            s = signal[p - 25:p + 35]
            std = np.array(s).std()
            if std < 17:
                new_peaks.append(p)
        return new_peaks