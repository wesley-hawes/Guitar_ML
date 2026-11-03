import copy
import unittest

import numpy as np

from note_events import group_pitch_windows


def frames(frequencies, accepted=None, origin=0):
    if accepted is None:
        accepted = [True] * len(frequencies)
    return [dict(start_sample=origin + i * 960, stop_sample_exclusive=origin + i * 960 + 4800,
                 sample_rate=48000, candidate_hz=hz, accepted=keep)
            for i, (hz, keep) in enumerate(zip(frequencies, accepted))]


class NoteEventTests(unittest.TestCase):
    def test_pitch_changes_and_final_singleton_are_preserved(self):
        events = group_pitch_windows(frames([146.4, 147.0, 164.5, 164.8, 220.0]), 960)
        self.assertEqual([e['note_name'] for e in events], ['D3', 'E3', 'A3'])
        self.assertEqual([e['window_count'] for e in events], [2, 2, 1])
        self.assertAlmostEqual(events[0]['median_hz'], 146.7)
        self.assertEqual(events[-1]['first_midpoint_seconds'], events[-1]['last_midpoint_seconds'])

    def test_rejected_window_splits_same_pitch_even_if_source_spans_overlap(self):
        events = group_pitch_windows(frames([110, np.nan, 110], [True, False, True]), 960)
        self.assertEqual(len(events), 2)
        self.assertEqual([e['first_frame_index'] for e in events], [0, 2])
        self.assertLess(events[0]['last_midpoint_seconds'], events[1]['first_midpoint_seconds'])
        self.assertGreater(events[0]['source_stop_sample_exclusive'], events[1]['source_start_sample'])

    def test_missing_window_breaks_run_and_input_is_unchanged(self):
        rows = frames([110, 110, 110], origin=48000)
        del rows[1]
        original = copy.deepcopy(rows)
        events = group_pitch_windows(rows, 960)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]['first_midpoint_seconds'], 1.05)
        self.assertEqual(rows, original)

    def test_gradual_drift_crosses_note_boundary_instead_of_chaining_forever(self):
        frequencies = 110 * 2 ** (np.array([0, 30, 60, 90]) / 1200)
        events = group_pitch_windows(frames(frequencies), 960)
        self.assertEqual([e['note_name'] for e in events], ['A2', 'A#2'])

    def test_labels_are_ignored_and_uninterrupted_same_pitch_stays_one_run(self):
        rows = frames([110] * 4)
        for i, row in enumerate(rows):
            row['interior_label_hz'] = 440 + i * 100
        self.assertEqual(len(group_pitch_windows(rows, 960)), 1)
        self.assertEqual(group_pitch_windows(rows, 960)[0]['note_name'], 'A2')

    def test_empty_rejected_and_invalid_inputs(self):
        self.assertEqual(group_pitch_windows([], 960), [])
        self.assertEqual(group_pitch_windows(frames([np.nan], [False]), 960), [])
        with self.assertRaises(ValueError):
            group_pitch_windows(frames([np.nan]), 960)
        with self.assertRaises(ValueError):
            group_pitch_windows(frames([110]), 0)
        rows = frames([110, 110]); rows[1]['start_sample'] = 0
        with self.assertRaises(ValueError):
            group_pitch_windows(rows, 960)


if __name__ == '__main__':
    unittest.main()
