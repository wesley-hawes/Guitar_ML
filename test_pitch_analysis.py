import unittest

import numpy as np

from pitch_analysis import estimate_period


class PitchAnalysisTests(unittest.TestCase):
    def test_known_tones_with_stronger_second_harmonics(self):
        for rate in (44100, 48000):
            time = np.arange(round(0.1 * rate)) / rate
            for frequency in (82.41, 110.0, 261.63):
                with self.subTest(rate=rate, frequency=frequency):
                    signal = (np.sin(2 * np.pi * frequency * time)
                              + 2 * np.sin(2 * np.pi * 2 * frequency * time))
                    measured, lag, score = estimate_period(signal, rate)
                    self.assertLess(abs(measured / frequency - 1), 0.01)
                    self.assertAlmostEqual(measured, rate / lag)
                    self.assertGreater(score, 0)
                    self.assertLessEqual(score, 1)

    def test_gain_and_dc_offset_preserve_the_selected_period(self):
        rate = 48000
        time = np.arange(4800) / rate
        signal = np.sin(2 * np.pi * 110 * time)
        original = estimate_period(signal, rate)
        changed = estimate_period(0.07 * signal + 0.2, rate)
        self.assertEqual(original[1], changed[1])
        self.assertAlmostEqual(original[2], changed[2])

    def test_constant_audio_has_no_pitch_candidate(self):
        for value in (0.0, 0.5):
            self.assertTrue(np.all(np.isnan(estimate_period(np.full(4800, value), 48000))))

    def test_invalid_audio_or_search_range_raises(self):
        for signal in (np.zeros((4800, 2)), np.full(4800, np.nan), np.zeros(10)):
            with self.subTest(shape=signal.shape), self.assertRaises(ValueError):
                estimate_period(signal, 48000)
        with self.assertRaises(ValueError):
            estimate_period(np.zeros(4800), 48000, 1000, 60)


if __name__ == "__main__":
    unittest.main()
