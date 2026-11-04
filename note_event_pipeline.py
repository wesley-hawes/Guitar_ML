# %% [markdown]
# # Combined note-entry table: pitch, rises, and quieter transitions
# This is the new integrated detection path. Original pitch calculations and
# rise-only experiment 09 remain reproducible. Run experiment 08 first.
# Each existing pitch group retains a row. Nearby rise/drop evidence describes
# its start; both cues for the same entry consolidate into ONE row. Additional
# rises inside a sustained group propose repeats and are explicitly flagged.
# No target counts, known phrase, filenames or expected labels guide assembly.
#
# Important: these are provisional note events, not finished transcription.
# End times remain blank. Evidence spans are analysis support, not durations.
# A wrong pitch group can still create a wrong note. An extra rise can create a
# false repeat, so inspect flagged repeats rather than calling counts accuracy.

# %%
from pathlib import Path
import csv
import hashlib
import json
from event_detection import build_note_events

folder = Path('outputs/pitch_tracking/personal_2026-11-03')
session = json.loads(Path('data/recording_sessions/2026-11-03_personal_takes.json').read_text())
if not (folder/'audit.json').exists():
    raise FileNotFoundError('Run python personal_recordings_lesson.py first.')
audit = json.loads((folder/'audit.json').read_text())
output = folder/'note_events';output.mkdir(exist_ok=True)
settings = {k:v for k,v in session['attack_candidate_settings'].items() if k!='status'}
integer_fields = {'sample_rate','start_sample','stop_sample_exclusive','history_start_sample',
                  'history_stop_sample_exclusive','rms_window_samples','hop_samples','history_samples',
                  'event_id','note_number','window_count','first_frame_index','last_frame_index',
                  'source_start_sample','source_stop_sample_exclusive'}
def read_rows(path):
    with path.open() as handle:
        rows=list(csv.DictReader(handle))
    for row in rows:
        for key,value in row.items():
            try: row[key]=int(value) if key in integer_fields else float(value)
            except ValueError: pass
    return rows

# %% [markdown]
# ## Run the same integration on all five recordings
# Association tolerance is 100 ms, reusing the pitch-window duration. This is a
# simple initial choice, not a tuned optimum. Too wide a tolerance can merge
# rapid repeats or attach an attack to the wrong entry. Keep this exposed.
# Default rise thresholds remain 3x / 1.5x / 3x, separation 80 ms.

# %%
results={}
report=['# Provisional note-entry tables', '',
        'Start times are estimates; note endings are not inferred. Evidence reasons are not confidence scores.',
        'Rows marked review-repeat may be amplitude fluctuations rather than actual re-picks.', '']
for item in session['recordings']:
    key=item['id'];path=Path(item['path'])
    if hashlib.sha256(path.read_bytes()).hexdigest()!=audit[key]['sha256']:
        raise ValueError('Audio changed; rerun experiment 08.')
    groups=read_rows(folder/f'{key}_note_evidence.csv')
    energy=read_rows(folder/f'{key}_energy_rise.csv')
    reference=read_rows(folder/f'{key}_windows.csv')[0]['reference_rms']
    events,evidence=build_note_events(groups,energy,reference,attack_settings=settings)
    with (output/f'{key}_events.csv').open('w',newline='') as handle:
        fields=list(events[0]) if events else ['event_id','note_name','estimated_start_seconds','end_seconds','reason']
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(events)
    (output/f'{key}_audit.json').write_text(json.dumps(dict(evidence=evidence,
        source_sha256=audit[key]['sha256'],reference_rms=reference,association_seconds=.1,
        attack_settings=settings),indent=2)+'\n')
    results[key]=events
    unassigned=sum(a['assignment']=='unassigned_no_pitch_support' for a in evidence['rise_assignments'])
    print(f'\n{key}: {len(groups)} pitch groups → {len(events)} proposed events; {unassigned} rises without pitch support')
    print(' ID  Note  Start(s)   Reason                 Review repeat')
    report += [f'## {key}', '', '| ID | Note | Estimated start (s) | Evidence | Review repeat |',
               '| --- | --- | --- | --- | --- |']
    for e in events:
        print(f"{e['event_id']:3}  {e['note_name']:4}  {e['estimated_start_seconds']:7.3f}    {e['reason']:22} {e['review_repeat']}")
        report.append(f"| {e['event_id']} | {e['note_name']} | {e['estimated_start_seconds']:.3f} | {e['reason']} | {'yes' if e['review_repeat'] else ''} |")
    report.append('')
(output/'note_events.md').write_text('\n'.join(report)+'\n')
print('\nSaved tables:',output)

# %% [markdown]
# ## How to read the table
# `pitch+rise`: level increase supports an entry. `pitch+drop`: changed pitch
# plus sharp level decrease recovers a quieter entry. `pitch+rise+drop` means
# both cues describe one group entry, not two notes. `pitch_only` preserves a
# note whose attack is unresolved. `pitch+rise_repeat` is a proposed re-pick
# within one group; review it because an amplitude fluctuation can imitate one.
#
# Start is a candidate measurement midpoint (or first pitch midpoint), not an
# exact physical onset. End is deliberately unknown. Repeated entries can share
# a group's evidence span. Do not use evidence_until as a note-off timestamp.
#
# Next evaluate this fixed MVP on a new recording, including a few known repeats
# and loud-to-soft transitions. Record actual played notes separately. Check
# individual matches, misses and extras, not only totals. Avoid more special rules
# for isolated anomalies. Same-pitch quiet re-picks remain an important limitation.
