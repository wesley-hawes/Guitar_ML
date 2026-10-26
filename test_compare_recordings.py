import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from compare_recordings import analyze_file, detect_onset, measure_windows


class ComparisonTests(unittest.TestCase):
    def test_silence_padding_does_not_change_aligned_measurements(self):
        sr = 44100
        note = .1 * np.cos(2 * np.pi * 110 * np.arange(sr) / sr)
        starts = [sr + 137, 2 * sr + 79]
        results = []
        for start in starts:
            signal = np.concatenate([np.zeros(start), note])
            onset, _, _ = detect_onset(signal, sr)
            self.assertEqual(onset, start)
            results.append(measure_windows(signal, sr, onset, duration=.5))
        np.testing.assert_allclose([w["rms"] for w in results[0]],
                                   [w["rms"] for w in results[1]])
        self.assertEqual(results[0][0]["start_sample"], starts[0])
        self.assertEqual(results[1][0]["start_sample"], starts[1])

    def test_isolated_click_does_not_count_as_sustained_note(self):
        sr = 44100
        signal = np.zeros(2 * sr)
        signal[int(.8 * sr)] = .5
        start = sr + 137
        signal[start:] = .1 * np.cos(2 * np.pi * 82 * np.arange(len(signal) - start) / sr)
        self.assertEqual(detect_onset(signal, sr)[0], start)

    def test_silence_and_stationary_noise_have_no_onset(self):
        rng = np.random.default_rng(4)
        for signal in [np.zeros(88200), rng.normal(0, .001, 88200)]:
            with self.assertRaises(ValueError):
                detect_onset(signal, 44100)

    def test_noise_threshold_scales_with_recording_gain(self):
        sr = 44100
        signal = np.random.default_rng(5).normal(0, .0001, 2 * sr)
        start = sr + 137
        signal[start:] += .1 * np.cos(2 * np.pi * 110 * np.arange(len(signal) - start) / sr)
        onset, noise, threshold = detect_onset(signal, sr)
        louder_onset, louder_noise, louder_threshold = detect_onset(signal * 2, sr)
        self.assertEqual(onset, louder_onset)
        self.assertAlmostEqual(louder_noise, 2 * noise)
        self.assertAlmostEqual(louder_threshold, 2 * threshold)

    def test_rms_peak_and_sample_boundaries_at_two_sample_rates(self):
        for sr in [44100, 48000]:
            size = round(.05 * sr)
            signal = np.full(sr, .5)
            windows = measure_windows(signal, sr, 123, duration=.1)
            self.assertEqual(len(windows), 2)
            for i, window in enumerate(windows):
                self.assertEqual(window["start_sample"], 123 + i * size)
                self.assertEqual(window["stop_sample_exclusive"], 123 + (i + 1) * size)
                self.assertAlmostEqual(window["seconds_after_onset"], .025 + i * .05)
                self.assertAlmostEqual(window["rms"], .5)
                self.assertAlmostEqual(window["absolute_peak"], .5)

    def test_partial_window_is_dropped_and_short_recording_rejected(self):
        windows = measure_windows(np.ones(250), 1000, 0, duration=.23)
        self.assertEqual(len(windows), 4)
        self.assertEqual(windows[-1]["stop_sample_exclusive"], 200)
        with self.assertRaises(ValueError):
            measure_windows(np.ones(250), 1000, 100, duration=.23)

    def test_manual_onset_and_mono_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "A_note_dry_take01.wav"
            sf.write(path, np.full(9 * 44100, .1), 44100, subtype="FLOAT")
            result = analyze_file(path, onset_override=123)
            self.assertEqual(result["onset_method"], "manual")
            self.assertEqual(result["windows"][0]["start_sample"], 123)
            self.assertEqual(len(result["windows"]), 160)
            with self.assertRaises(ValueError):
                analyze_file(path, onset_override=-1)


if __name__ == "__main__":
    unittest.main()
