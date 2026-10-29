import tempfile
import unittest
from pathlib import Path

import mido

from guitartechs_midi import read_note_labels


class MidiLabelTests(unittest.TestCase):
    def read_tracks(self, tracks, file_type=1):
        midi = mido.MidiFile(type=file_type, ticks_per_beat=960)
        midi.tracks.extend(mido.MidiTrack(track) for track in tracks)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "labels.mid"
            midi.save(path)
            return read_note_labels(path)

    def test_tempo_changes_and_zero_velocity_note_off(self):
        labels, _ = self.read_tracks([
            [mido.MetaMessage("set_tempo", tempo=1000000),
             mido.MetaMessage("set_tempo", tempo=2000000, time=480)],
            [mido.MetaMessage("track_name", name="E"),
             mido.Message("note_on", note=40, velocity=80),
             mido.Message("note_on", note=40, velocity=0, time=960)],
        ])
        self.assertEqual(len(labels), 1)
        self.assertEqual(labels[0]["start_seconds"], 0)
        self.assertAlmostEqual(labels[0]["end_seconds"], 1.5)
        self.assertEqual(labels[0]["string_track"], "E")

    def test_same_channel_and_pitch_on_different_strings_stay_separate(self):
        labels, _ = self.read_tracks([
            [mido.MetaMessage("track_name", name="E"),
             mido.Message("note_on", note=45, velocity=70, channel=0),
             mido.Message("note_off", note=45, channel=0, time=960)],
            [mido.MetaMessage("track_name", name="A"),
             mido.Message("note_on", note=45, velocity=80, channel=0, time=480),
             mido.Message("note_off", note=45, channel=0, time=960)],
        ])
        self.assertEqual([n["string_track"] for n in labels], ["E", "A"])
        self.assertEqual([n["start_seconds"] for n in labels], [0, 0.25])
        self.assertEqual([n["end_seconds"] for n in labels], [0.5, 0.75])

    def test_unmatched_or_ambiguous_events_are_not_silently_dropped(self):
        cases = [
            [mido.Message("note_off", note=40)],
            [mido.Message("note_on", note=40, velocity=80)],
            [mido.Message("note_on", note=40, velocity=80),
             mido.Message("note_on", note=40, velocity=80, time=10)],
        ]
        for messages in cases:
            with self.subTest(messages=messages), self.assertRaises(ValueError):
                self.read_tracks([messages])

    def test_asynchronous_tracks_are_rejected(self):
        with self.assertRaises(ValueError):
            self.read_tracks([[mido.MetaMessage("end_of_track")]], file_type=2)


if __name__ == "__main__":
    unittest.main()
