"""Convert MIDI note events to label rows while preserving string-track names."""

from collections import Counter
import mido


def read_note_labels(path):
    midi = mido.MidiFile(path)
    if midi.type not in (0, 1) or midi.ticks_per_beat <= 0:
        raise ValueError("Use synchronous MIDI with ticks-per-quarter timing.")

    # Collect absolute tick positions. String identity comes from the track;
    # P1 uses the same MIDI channel on all six strings.
    events = []
    for track_index, track in enumerate(midi.tracks):
        tick = 0
        for event_index, message in enumerate(track):
            tick += message.time
            events.append((tick, track_index, event_index, message))
    events.sort(key=lambda event: event[:3])

    tempo = 500000  # MIDI default: microseconds per quarter note.
    previous_tick = 0
    seconds = 0.0
    active = {}
    notes = []
    tempo_events = []
    for tick, track_index, _, message in events:
        seconds += mido.tick2second(tick - previous_tick, midi.ticks_per_beat, tempo)
        previous_tick = tick
        if message.type == "set_tempo":
            if message.tempo <= 0:
                raise ValueError("MIDI tempo must be positive.")
            tempo = message.tempo
            tempo_events.append((seconds, tempo))
        elif message.type in ("note_on", "note_off"):
            key = (track_index, message.channel, message.note)
            if message.type == "note_on" and message.velocity > 0:
                if key in active:
                    raise ValueError("Overlapping repeated MIDI note: review pairing manually.")
                active[key] = (seconds, message.velocity, tick)
            else:
                # A note_on with velocity zero is also a note-off.
                if key not in active:
                    raise ValueError("Unmatched MIDI note-off.")
                start, velocity, start_tick = active.pop(key)
                if seconds <= start:
                    raise ValueError("MIDI note has no positive duration.")
                notes.append(dict(
                    start_seconds=start, end_seconds=seconds,
                    start_tick=start_tick, end_tick=tick,
                    midi_note=message.note, velocity=velocity,
                    string_track=midi.tracks[track_index].name,
                    track_index=track_index, channel=message.channel,
                ))
    if active:
        raise ValueError("Unclosed MIDI note-on.")
    notes.sort(key=lambda note: (note["start_seconds"], note["track_index"], note["midi_note"]))
    return notes, dict(ticks_per_quarter=midi.ticks_per_beat,
                       tempo_events=tempo_events,
                       notes_per_track=dict(Counter(n["string_track"] for n in notes)))
