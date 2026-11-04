import unittest

import numpy as np

from attack_analysis import measure_energy_rise, detect_attack_candidates


class EnergyRiseTests(unittest.TestCase):
    def test_past_blocks_and_original_sample_bounds(self):
        # Four two-sample blocks of known RMS; median(1,2,3,4)=2.5.
        signal = np.repeat([1., 2., 3., 4., 5.], 2)
        rows = measure_energy_rise(signal, 1000, origin_sample=100,
                                  window_seconds=.002, hop_seconds=.001, history_seconds=.008)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual((row['history_start_sample'], row['start_sample'], row['stop_sample_exclusive']), (100, 108, 110))
        self.assertEqual(row['midpoint_seconds'], .109)
        self.assertEqual(row['past_median_rms'], 2.5)
        self.assertEqual(row['positive_rise'], 2.5)
        self.assertEqual(row['rms_ratio'], 2)

    def test_renewed_level_does_not_require_prior_silence(self):
        signal = np.concatenate([np.full(2400, .1), np.full(2400, .2)])
        rows = measure_energy_rise(signal, 48000)
        step = next(row for row in rows if row['start_sample'] == 2400)
        self.assertAlmostEqual(step['past_median_rms'], .1)
        self.assertAlmostEqual(step['positive_rise'], .1)
        self.assertAlmostEqual(step['rms_ratio'], 2)
        self.assertTrue(all(abs(r['positive_rise']) < 1e-12 for r in rows if r['stop_sample_exclusive'] <= 2400))

    def test_future_samples_cannot_change_completed_measurements(self):
        rng = np.random.default_rng(7)
        original = rng.normal(size=6000)
        changed = original.copy(); changed[4000:] *= 20
        first = measure_energy_rise(original, 48000)
        second = measure_energy_rise(changed, 48000)
        for a, b in zip(first, second):
            if a['stop_sample_exclusive'] <= 4000:
                self.assertEqual(a, b)

    def test_gain_scales_absolute_rise_but_preserves_ratio(self):
        signal = np.concatenate([np.full(2400, .1), np.full(2400, .2)])
        first = measure_energy_rise(signal, 48000)
        second = measure_energy_rise(signal * 3, 48000)
        np.testing.assert_allclose([r['positive_rise'] for r in second],
                                   np.array([r['positive_rise'] for r in first]) * 3, atol=1e-15)
        np.testing.assert_allclose([r['rms_ratio'] for r in first], [r['rms_ratio'] for r in second])

    def test_silence_partial_tail_and_invalid_inputs(self):
        rows = measure_energy_rise(np.zeros(2401), 48000)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['positive_rise'], 0)
        self.assertTrue(np.isnan(rows[0]['rms_ratio']))
        self.assertEqual(measure_energy_rise(np.zeros(100), 48000), [])
        for signal in (np.full(2400, np.nan), np.zeros((2400, 2))):
            with self.assertRaises(ValueError):
                measure_energy_rise(signal, 48000)
        with self.assertRaises(ValueError):
            measure_energy_rise(np.zeros(2400), 48000, history_seconds=.035)


class AttackCandidateTests(unittest.TestCase):
    def test_ringing_control_vs_renewed_excitation_without_silence(self):
        rate = 48000
        t = np.arange(int(1.4 * rate)) / rate
        carrier = np.sin(2 * np.pi * 440 * t)
        old = .12 * np.exp(-.8*t) * carrier
        new = .08 * np.clip((t-.7)/.005, 0, 1) * np.exp(-.8*np.maximum(t-.7, 0)) * carrier
        control, _ = detect_attack_candidates(measure_energy_rise(old, rate), .0001)
        episodes, _ = detect_attack_candidates(measure_energy_rise(old+new, rate), .0001)
        self.assertEqual([e for e in control if e['selected']], [])
        selected = [e for e in episodes if e['selected']]
        self.assertEqual(len(selected), 1)
        self.assertTrue(.695 <= selected[0]['peak_midpoint_seconds'] <= .735)
        self.assertGreater(selected[0]['past_median_rms'], .03)

    def test_zero_baseline_initial_attack_and_quiet_pulse(self):
        # Huge ratio against digital zero must not admit a below-floor pulse.
        for amplitude, expected in ((.1, 1), (.0001, 0), (0, 0)):
            signal = np.zeros(500)
            signal[100:200] = amplitude
            episodes, decisions = detect_attack_candidates(measure_energy_rise(signal, 1000), .001)
            self.assertEqual(sum(e['selected'] for e in episodes), expected)
            self.assertTrue(all(np.isfinite(r['rise_ratio']) for r in decisions))

    def test_calibrated_gain_preserves_decisions(self):
        signal = np.full(500, .01);signal[100:200] = .1;signal[300:400] = .04
        a, da = detect_attack_candidates(measure_energy_rise(signal, 1000), .001)
        b, db = detect_attack_candidates(measure_energy_rise(signal*3, 1000), .003)
        self.assertEqual([e['peak_start_sample'] for e in a if e['selected']],
                         [e['peak_start_sample'] for e in b if e['selected']])
        self.assertEqual([r['qualifies'] for r in da], [r['qualifies'] for r in db])

    def test_stronger_neighbor_suppression_and_final_episode(self):
        rows = [dict(start_sample=i*10, stop_sample_exclusive=i*10+10,
                     sample_rate=1000, hop_samples=10, current_rms=x, past_median_rms=1.)
                for i,x in enumerate([1.,2.,2.5,1.,4.])]
        episodes, _ = detect_attack_candidates(rows, .01)
        self.assertEqual(len(episodes), 2)
        self.assertEqual(episodes[0]['qualifying_frame_count'], 2)
        self.assertEqual(episodes[0]['peak_start_sample'], 20)
        self.assertFalse(episodes[0]['selected'])
        self.assertEqual(episodes[0]['suppression_reason'], 'within_separation_of_episode_2')
        self.assertTrue(episodes[1]['selected'])  # final episode flushed

    def test_origin_is_preserved_and_missing_windows_split(self):
        rows = [dict(start_sample=i, stop_sample_exclusive=i+10, sample_rate=1000,
                     hop_samples=10, current_rms=2., past_median_rms=1.) for i in [100,110,300]]
        episodes, _ = detect_attack_candidates(rows, .01)
        self.assertEqual(len(episodes), 2)
        self.assertEqual(episodes[0]['first_qualifying_start_sample'], 100)
        self.assertEqual(episodes[0]['peak_midpoint_seconds'], .105)
        self.assertTrue(all(e['selected'] for e in episodes))

    def test_invalid_reference_settings_and_rows(self):
        self.assertEqual(detect_attack_candidates([], .01), ([], []))
        for ref in (0, -1, np.nan):
            with self.assertRaises(ValueError): detect_attack_candidates([], ref)
        for settings in ({'minimum_rise_ratio':1}, {'minimum_separation_seconds':0}):
            with self.assertRaises(ValueError): detect_attack_candidates([], .01, **settings)
        rows=measure_energy_rise(np.ones(100), 1000)
        with self.assertRaises(ValueError): detect_attack_candidates(rows[::-1], .01)
        rows[0]['current_rms']=np.nan
        with self.assertRaises(ValueError): detect_attack_candidates(rows, .01)


if __name__ == '__main__':
    unittest.main()
