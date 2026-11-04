# Guitar ML — project handoff

Last updated: 2026-11-03. Read this alongside README.md and the actual code.
Verify the current files and Git status before making changes; this document is
a checkpoint, not a substitute for inspecting the project.

## Resume here — first attack detector and Fastest take (2026-11-03)

User confirms fifth-position A minor pentatonic from D-string A to low-E A,
consistent with D7 A3 through low-E5 A2 in standard tuning. C-to-A's 20 plucks
are correct (5 C, 5 A, repeated). User cannot hear anything before intended A
in Dynamics; early E2 is unwanted relative to intended phrase, physical cause
unresolved. Do not claim it is definitely hum or a played low E. Gain/export
consistency and candidate quiet interval still not explicitly confirmed.

Added Fastest WAV to manifest and notebook 08/script. Unchanged 100 ms/20 ms
pitch pipeline gives intended 12-note sequence plus B2 at 4.25–4.27 s, 75/312
passing frames and 13 groups. Intended first/final group starts 1.15/3.11 s,
span 1.96 s versus baseline 4.10 s (~52% shorter). Several intermediate notes
have only 3–9 accepted frames; no rule tuned to recover them or remove B2.

New `detect_attack_candidates` in attack_analysis.py returns (episodes,
frame_decisions). It consumes existing energy measurements, not pitch/labels.
Frozen initial defaults recorded in session JSON before viewing fastest or
candidate results: current >=3x reference; current/max(past median,reference)
>=1.5; positive rise >=3x reference. Consecutive qualifying rows form episodes,
peak positive rise represents each. Strongest-first suppression keeps peaks
>=80 ms apart; chronological episodes retain selected/reason flags. No minimum
sustained duration. Offline lookahead, not streaming. First 40 ms has no candidate
measurements. Timing is measurement midpoint, not exact physical onset.

New `09_detecting_attacks.ipynb` / `attack_detection_lesson.py`: reuse 08 CSVs,
check WAV hashes, export all decisions/episodes/settings to ignored
`outputs/pitch_tracking/personal_2026-11-03/attack_candidates/`. Run 08 before 09
if outputs missing/stale. P2 check optionally reuses earlier 06/07 measurements.
Pitch/grouping unchanged; no candidate-driven event assembly yet.

Initial attack results (counts NOT accuracy): baseline 12/14 selected/episodes;
dynamics 12/16; faster 12/20; CtoA 20/21; fastest 13/17. CtoA selected markers
visually align with 20 attacks. Earlier P2 excerpt proposes 7.991, 8.471, 8.969 s:
two repeated A4s plus following G4, whereas pitch-only A4 grouping stayed merged.
Failures: baseline misses soft E3 entry around 3.3 s, adds an extra later D3
candidate. Dynamics includes early unwanted sound and misses soft E3 entry.
Fastest misses E3 entry around 2.05 s and adds late candidates 3.415/4.231 s.
Max rise ratios in reviewed soft transitions: baseline 3.2–3.45 s ~1.286;
dynamics 3.5–3.8 s ~1.165; fastest 2.00–2.16 s ~1.195, all below 1.5. Do not lower
threshold merely to force expected count. No manually timed labels or listening
by agent, so interpret misses/extras as visual pitch/energy comparison.

32 tests pass, including six new detector checks: no-silence renewed excitation
versus decay control, zero baseline/noise floor, gain/reference scaling,
stronger-neighbor suppression/final flush, original bounds/missing measurements,
invalid inputs. Real audio results are validation evidence distinct from tests.
No new dependencies. Original WAVs unchanged. No commit/push requested.

Next: user reads 09 sections 2–3; review/annotate a handful of pluck intervals,
including repeated C and a soft E3. Then choose how pitch and attack evidence
combine into note events. Do not require accepted pitch at exact attack instant
(similarity often drops then), and do not lose every pitch change without energy
rise (legato is real). Possible future spectral-change cue if amplitude alone
keeps missing soft picks; no immediate ML/chord expansion. Separate onset,
offset, continuous pitch, event assembly, and string/fret inference. All five
takes now inspected: use a NEW take for future held-out evaluation.

Asked learning question asynchronously: should reliable pitch be required at
the attack instant or looked for after an independently detected attack, and
what is the tradeoff? Await user's reasoning; continue explaining architecture.

## Previous checkpoint — four personal takes analyzed (2026-11-03)

Newest lesson: `08_personal_recordings.ipynb`, generated from the percent-cell
script `personal_recordings_lesson.py`. Session metadata lives in
`data/recording_sessions/2026-11-03_personal_takes.json`. User reports neck + middle
pickup, pretty dry tone, no metronome. Do NOT carry the earlier neck-only/9 o'clock
gain forward as confirmed for this batch. Gain/export consistency, exact string/
fret positions, phrase repetitions and repeated C/A counts were asked but are
still unconfirmed. Four input WAVs are unchanged, 44.1 kHz/16-bit identical-channel
stereo, finite, with no full-scale samples. Numerical/plot inspection, no listening.

Current pitch estimator, acceptance and grouping unchanged: 100 ms windows,
20 ms hops, 60–1000 Hz; level >=3x reference, similarity >=.80, adjacent eligible
estimates within 50 cents. Energy-rise helper remains inspection-only. Do NOT
describe these groups as attack-detector output. Intended sequence never enters
prediction. Full files analyzed, no trimming to expected phrase or note snapping.

Reference is provisional: DynamicsAdded 1.2–1.4 s, six complete overlapping
100 ms windows on original 20 ms grid, median RMS ~0.00002170. User confirmation
that it was quiet and gains stayed consistent is pending. Other files have
exact-zero padding; using it as background would create a misleading zero floor.
Reference calibration remains a limitation, not a universally validated gate.

Results (support spans are window midpoints, NOT exact physical onsets/offsets):
- Baseline: 232/367 windows pass, 12 groups A3 G3 E3 D3 G3 E3 D3 C3 E3 D3 C3 A2.
- Faster: 370/441 pass, same 12 groups. First-to-final group starts span 3.28 s
  versus baseline 4.10 s (~20% shorter); not yet an extreme fast-picking test.
- Dynamics: 403/515 pass, same phrase plus E2 at 0.81–1.09 s before it. Could be
  a ringing low string or unwanted periodic sound; review before calling it a
  false positive. No special threshold change/removal for this example.
- CtoA: 642/813 pass, 20 groups: 5 C3, 5 A2, 5 C3, 5 A2. This is consistent with
  visible repeated attacks but actual intended pluck count needs confirmation.
  Similarity dips/rejected windows split runs; energy cues do NOT cause splits.
  Prior P2 repeated A4 still demonstrates that repeats can merge in this design.
No timing accuracy or percent-correct score: no independent timestamp/octave
labels. Long tails dominate passing-frame counts. All four are development data
now that inspected. Save a new future take as held-out evaluation.

Outputs: ignored `outputs/pitch_tracking/personal_2026-11-03/` contains four
overview plots, repeated-C close-up, reference review, per-window/event/energy
CSVs, and input SHA256 audit. At 44.1 kHz, pitch width/hop=4410/882 samples;
energy width/history/hop=441/1764/88 samples (2 ms rounded to ~1.995 ms). All
times derive from original indices. Every accepted frame belongs to one group.
Validation: five notebook code cells executed, notebook schema valid, plots
inspected; 2,136 acceptance decisions checked against existing rules, sample
bounds and sampled RMS/history calculations checked. Input hashes and older
CSV hashes unchanged. No analysis-core changes or new dependencies.

Next: user reviews notebook 08 and confirms recording details; listen/review
early Dynamics E2, then annotate a few repeated-pluck onset intervals. Design
an explicit attack-candidate rule before splitting groups. Do not move directly
to chords or ML merely because these single-note pitch classes agree. Soft
attacks may have small absolute rises; don't set a threshold from loud accents.
Later bends/slides need continuous pitch rather than treating every nearest-note
change as a new pluck. Pitch alone does not uniquely determine string/fret.

User renewed preferences: challenge assumptions and ask short learning questions
about architecture/design; explain decisions before piling on code. Keep eventual
tablature in view. Prefer Medium; no current-thread model-switch tool was exposed
in this session, so no switch was claimed/performed. Do not spawn another thread
just to simulate switching. No commit/push requested for this analysis.

## Previous checkpoint — renewed-energy cue implemented (2026-11-03)

User authorized the energy-rise calculation and exploring picking while a note
is already ringing. Added 07_understanding_attacks.ipynb, attack_lesson.py,
attack_analysis.py and test_attack_analysis.py. Still a measurement/inspection
lesson: no attack threshold, automatic attack list or event splitting added.

Pure helper measure_energy_rise: current 10 ms RMS vs median RMS of FOUR prior
nonoverlapping 10 ms blocks (40 ms history), advanced every 2 ms. History ends
exactly at current-window start, so no current/future samples enter its baseline.
Return current RMS, past median, signed change, max(0, change), and RMS ratio;
zero baseline gives NaN ratio. Raw amplitudes, no silence prerequisite, no MIDI.
RMS is an amplitude measure, not physical energy. Original sample bounds and
history bounds retained; plotted midpoint needs audio through window end.

Lesson compares P1/P2 7.8–9.1 s, reading an extra 40 ms of preceding context.
Each has 646 measurements. P2 largest absolute rise near second A4: 0.021983
at midpoint 8.471 s, recent RMS 0.000815, ratio 27.99. Ordinary ringing in
8.1–8.35 s has max rise 0.000229. P1 corresponding rise 0.040869 at 8.491 s;
ordinary comparison max 0.000487. These are maxima inside reviewed intervals,
not detected attacks, accuracy scores, or threshold suggestions. Rise magnitudes
vary with gain; ratios can exaggerate tiny noise over near-zero baselines.

Synthetic illustration (not guitar validation): decaying 440 Hz signal receives
another in-phase 440 Hz excitation at 0.7 s, ramped over 5 ms; old ringing never
muted. Same method shows rise 0.055907 over nonzero baseline 0.049037, ratio 2.14;
the old-ring-only control has zero positive rise in that review interval. Phase-
aligned addition is deliberately simplified: real re-picking may damp/cancel or
change timbre without clear RMS growth. Actual no-gap re-picking remains untested.

Outputs under ignored outputs/pitch_tracking: P1/P2_scale_C_energy_rise_7p8-9p1s.csv,
P1_P2_repeated_A4_energy_rise.png and synthetic_renewed_A4_energy_rise.png. Original
pitch/event CSV hashes unchanged. Four notebook cells executed directly with
project Python; schema valid; both plots inspected. All 26 tests pass (5 new
checks for past-only history, original sample indices, renewed level without
silence, gain scaling, silence/invalid/partial inputs). No new dependencies.

Next: user reads lesson 07, especially real versus synthetic plots. Then design
an explicit candidate rule with enough signal, relative/absolute rise, and a
short separation rule to avoid duplicate attacks; do not require a quiet gap.
Review candidates before using them to split note groups. Actual quiet/fast
re-picking may need spectral-change evidence later. No commit/push requested.

### User model-usage preferences (added during this task)

Prefer Astra Medium for roughly 80–90% of development; conserve High usage.
Reserve High recommendations for complex architecture, difficult debugging,
advanced ML reasoning, or Medium struggling, and explain the benefit first.
Do not recommend High for routine implementation/refactoring/UI work. Minimize
redundant calls, excessive context reads and unnecessary tokens. Near usage
limits, finish the active task before starting more. Never claim to switch
models unless the environment supports it; otherwise recommend only as needed.

## Previous checkpoint — repeated A4 waveform plotted (2026-11-03)

User requested P2's repeated waveform. Notebook 06/script now default to
player='P2', excerpt_seconds=9.3. Added section 7: raw waveform over 7.8–9.1 s,
10 ms RMS / 2 ms hops alongside existing 100 ms RMS, and a waveform close-up
8.40–8.56 s. Dashed markers are supplied A4 MIDI starts, not detected attacks.
These extra energy windows are for inspection ONLY; estimator/filter/grouping
settings and decisions unchanged. Raw amplitudes preserved, no clip normalization.

Observation: waveform dips around 8.42–8.45 s then rises sharply around 8.46 s,
while pitch stays near A4. Short-window RMS reveals the dip; old 100 ms RMS
smooths across it. Renewed energy is a candidate cue for another pluck, not an
automatically measured onset. MIDI second start ~8.482 s is later than the
visible rise. Do not tune a threshold or correct timestamps from this example.
Agent inspected plots/numbers, not listening audio. No attack detector added.

Outputs: P2_scale_C_repeated_A4_waveform.png and
P2_scale_C_repeated_A4_10ms_rms.csv under ignored outputs/pitch_tracking.
The inspected interval is original samples [374400:436800] at 48 kHz. 646
complete energy windows, width 480 samples, hop 96; every RMS/sample bound/
midpoint checked against original WAV. Eight notebook cells executed directly
with project Python; schema valid, plot viewed, preexisting CSV hashes unchanged.

Next: review section 7, explain why finer energy windows reveal a new pluck
despite unchanged pitch. Then compare P1/another transition before designing a
small explained attack rule. Keep pitch settings fixed. No commit/push requested.

## Previous checkpoint — P1 scale transfer checked; repeated A4 selected (2026-11-03)

User asked to run the P1 transfer and report results before attack detection.
Parameterized scale_tracking_lesson.py and notebook 06 with player='P1' by
default; use 'P2' to reproduce prior results. Excerpt defaults to 8 s; output
filenames now include player and actual duration. Exact original P1/P2 noise
reference intervals retained. No estimator, threshold or grouping changes.
Extracted only P1 scale C DI WAV and its MIDI into Downloads/P1_scales.

P1 first 8 s: 396 windows, 316 accepted, 14 evidence groups A2 through G4.
All 109 guarded interior windows pass and differ <=50 cents from their labels.
All 14 groups match the 14 labels in pitch order and overlap their intervals;
no extra fragmentation/neighboring-pitch merging observed in this short excerpt.
P1 first 0.9 s is exact digital zero and MIDI starts at A2 ~1.023 s, whereas
P2 contains F2/G2 beforehand. The difference from P2's 16 groups is not two
missed P1 notes. Reference RMS is 0.0001149 from the same 2 prior P1 windows.

Extended both excerpts to 9.3 s with the same settings to select an attack case:
- P1 A4 labels 8.007–8.472 and 8.502–8.935 s; evidence groups 7.97–8.41 and
  8.47–8.91 s. At 8.43 weak repetition rejects; 8.45 waits for predecessor.
- P2 A4 labels 7.998–8.478 and 8.482–8.933 s; ONE evidence group 7.97–8.91 s,
  48 accepted windows. Pitch/acceptance stay continuous through the second label.
This is the concrete next example: P2 C scale, ~7.8–9.1 s, repeated A4. MIDI
suggests repeated picking; listen/inspect the waveform before claiming exact
physical attacks. No attack detector or chord separation was implemented.

Notebook final markdown records transfer results and reproduction settings:
player='P2', excerpt_seconds=9.3, run all cells. Both players' 0–9.3 s windows,
events and plots saved separately in ignored outputs/pitch_tracking. P2 0–8 s
window/event CSVs reproduce byte-for-byte after parameterization. P1 carried
reference exact, 396 bounds/hops and all accepted memberships checked. Seven
notebook cells executed directly with project Python, schema valid, plot viewed.
No dependencies added; unchanged core tests weren't needlessly rerun.

At start, Git was clean and main matched cached origin/main after prior commit
78be7fe (not a fresh remote verification). This transfer is a new uncommitted
change; user requested a report, not another commit/push.

Next: learn/plot an attack cue for P2's merged A4 pair, keeping pitch thresholds
fixed. Use P1's already split pair as a useful comparison. Preserve the learning
pace; avoid broad refactoring, more offset tuning, or moving straight to chords.

## Previous checkpoint — pitch windows grouped into evidence runs (2026-11-03)

Wrap-up: user approved the result and requested a meaningful commit and GitHub
push for all pending lesson/filter/grouping work. Reviewed the ten changed/new
project files; generated outputs and external audio remain excluded. All 21
tests passed in the completed grouping checks, and git diff --check is clean.
Intended commit: "Add pitch filtering and group scale estimates into note evidence".
The fetch attempt failed with "Could not resolve host: github.com" in this tool
environment. Check actual Git log/status and push output for the final result;
do not equate a local commit or cached origin/main with a successful upload.

User authorized grouping accepted estimates into note intervals, retaining
uncertainty around transitions. Added sections 5–6 to notebook 06 and its script,
plus pure function group_pitch_windows in note_events.py. The estimator and
acceptance thresholds are unchanged; existing window CSVs remain unchanged.

Grouping: map Hz to nearest equal-tempered note via
floor(69 + 12*log2(hz/440) + 0.5), A4=440. Join consecutive accepted windows
only while the assigned note stays the same. Rejected rows, missing-hop rows,
and pitch-name changes split runs. No gap filling, duration cutoff, pruning,
or MIDI-derived decisions. Keep singleton runs and the final run. Raw median
Hz is retained, not snapped to the nominal note. Grouping only receives audio
fields explicitly extracted from rows, not MIDI-derived comparison fields.

Result: 355 accepted frames become 16 evidence runs, F2 G2 A2 B2 C3 D3 E3 F3
G3 A3 B3 C4 D4 E4 F4 G4. These are not confirmed physical note events or an
accuracy score. D3 midpoint span 2.51–2.93 s; E3 3.01–3.43 s. Rejected gap
stays visible. Their underlying source sample spans overlap by 20 ms because
the analysis windows overlap; this is not evidence of simultaneous sound.

Table and plot show first/last supporting window MIDPOINTS, not inferred onset,
offset, or duration. CSV also includes first/last frame indices, window count,
median frequency, note number/name, source sample bounds, and sample rate.
Separate output files: P2_scale_C_0-8s_note_evidence.csv / .png under ignored
outputs/pitch_tracking. Gray crosses show rejected raw estimates; green bars
are grouped midpoint spans; original MIDI intervals are a separate lower panel.
Brief D#4 label and excerpt-edge A4 remain in the comparison; no groups were
fabricated to match them. All 355 accepted frames are assigned exactly once.

Validation: 6 new test_note_events tests, 21 repository tests total all pass.
Covers pitch change/final singleton, rejected gaps despite source-span overlap,
missing windows, original timeline offsets, nonmutation, gradual drift across
note bins, independence from MIDI fields, empty/invalid inputs. Seven notebook
cells executed directly with project Python, schema valid, plot inspected.
Original CSV hashes unchanged; frame membership and midpoint/source bounds
checked against scale rows. No new dependencies or commit/push requested.

Next: user reviews section 5's rule and one group's member rows, then section 6's
plot. Try a passage with repeated pitches to show what attack detection must add.
Same-note plucks without rejection may merge; rejection can fragment one played
note; bends/vibrato can cross bins. No confidence percentages, acoustic overlap
separation, exact timing, or tablature/string assignment has been added.

## Previous checkpoint — P2 scale transitions checked (2026-11-03)

User chose scales next and expects overlap between notes to be difficult. Added
06_tracking_a_scale.ipynb and scale_tracking_lesson.py. Extracted only
P2_scales/audio/directinput/directinput_C.wav and P2_scales/midi/midi_C.mid
from Downloads/P2_scales.zip into Downloads, preserving paths. No third-party
audio copied into repo. Scale WAV is 160 s, 48 kHz, stereo PCM24; analyze first
8 s only. Channels identical in excerpt. Filename C, but sequence starts F2;
use actual labels, not assumed root. Main ascent reaches G4, with A4 label
starting ~7.998 s at the edge. Brief D#4 label (~19 ms near 6.93 s) retained.

Settings unchanged: 100 ms windows / 20 ms hop, 60–1,000 Hz estimator, raw level
>=3x reference, similarity >=0.80, successive eligible pitches <=50 cents apart.
Scale begins with notes, so carried EXACT P2 single-note raw-RMS reference
0.0002565 from 7.70–7.86 s in the original single-note WAV. This is explicitly
provisional cross-file calibration; same player/source does not guarantee same
noise or gain. No scale-specific silence assumption, offset fix or tuning.

Results: 396 windows, 355 pass. 119 guarded MIDI-interior windows all pass and
are <=50 cents from reference. Guard requires whole window inside one label
with 100 ms margins. This is not full transcription accuracy. 42 windows
intersect multiple label spans; 21 pass a single pitch. These windows may
straddle sequential notes without actual simultaneous sound. Physical overlap
has not been measured or separated. Estimator still outputs one frequency.

Zoom 2.75–3.35 s: MIDI D3 (A string) → E3 (D string). Shaded [2.94:3.04] s
window midpoint 2.99 intersects both labels with a gap between. Midpoint 2.93:
147.7 Hz / score .824 passes; 2.95: 155.8 Hz / .427 rejected; 2.99: 164.4 Hz /
.869 waits for predecessor; 3.01: 164.4 Hz / .909 passes. These are window
midpoints, NOT attack times. 3.01 covers [2.96:3.06] s and includes audio after
MIDI start ~3.019. Keep annotation timing uncertainty visible; no correction.

Verification: original CSVs unchanged by notebook execution, five cells run
directly with project Python, notebook schema validated, plot viewed. Scale
decisions/reasons match lesson 04's original rule for all 396 rows; exact carried
reference verified, frame bounds/midpoints and reported comparisons checked.
Outputs: outputs/pitch_tracking/P2_scale_C_0-8s_transitions.png and
P2_scale_C_0-8s_windows.csv (ignored). No new dependencies. No commit/push.

Next: review notebook 06's overview and transition table, then group passing
windows into rough note events with uncertain gaps visible. Repeated same-pitch
picking needs attack detection later. Do not treat passing windows spanning
labels as successful separation of overlapping pitches. Chords remain future
work; no more tuning around 318 Hz or faint note ends.

## Previous checkpoint — broader check completed, settings unchanged (2026-11-03)

User asked to check whether the weak/offset-dominated pattern occurs elsewhere,
explicitly avoiding overfitting. Added 05_checking_more_audio.ipynb and matching
broader_pitch_check.py. They inspect first 30 s of BOTH P1/P2 with unchanged
100 ms windows, 20 ms hops, 60–1,000 Hz search, same manual raw-RMS references,
and same 3x / 0.80 / 50-cent acceptance rule. Lesson 04 is preserved. New tagged
CSV/PNG files live in ignored outputs/pitch_tracking; original CSVs unchanged.

New measurements only: window mean, centered RMS, mean**2 / raw_RMS**2. Verified
raw_RMS**2 = centered_RMS**2 + mean**2. More than 0.5 is a REVIEW flag indicating
majority mean contribution, NOT a filter, truth label, or proof of noise. The
centered RMS plot uses the same raw background denominator for comparison; no
recalibrated alternative detector or preprocessing change was applied.

Results per player: 1,496 overlapping windows, eight MIDI note spans (last is
partial). P1 passes 1,247; P2 passes 1,311. P1 has zero majority-mean windows.
P2 has 106, including 67 after the previous 11.9 s excerpt. All 67 new ones are
rejected by the existing rule. The only passing majority-mean window remains
3.81 s / 317.88 Hz. Thus the offset pattern recurs, but no additional passing
instance was found by this diagnostic. This does not rule out other failure types.

MIDI comparison uses complete audio windows inside label intervals with 100 ms
guards at both ends. P1 passes 1,171 / 1,249 such windows (78 rejected); P2 passes
all 1,207. All passing interior estimates differ from nominal MIDI by <=50 cents.
This is conditional interior agreement, not whole-recording accuracy: boundaries
are excluded and windows overlap. Do not hide the rejected P1 coverage. MIDI
enters only after prediction/acceptance, never to choose a candidate or filter.

Verification: all 591 existing windows/decisions per player reproduce; existing
CSV hashes unchanged. Checked 1,496 frame boundaries, hops, midpoints and fraction
bounds; executed four new notebook cells directly with project Python, validated
schema and visually inspected plot. No new dependencies or algorithm changes.
The small diagnostic repeats the explicit acceptance calculation from lesson 04;
old-window agreement is checked, but shared extraction/tests remain future cleanup.

Next: review notebook 05's plot and results. Keep current thresholds. Briefly
inspect some rejected P1 note interiors to understand coverage losses, then
advance toward note-event grouping rather than tuning this isolated 318 Hz case.
Only low single notes on these two recordings were checked; fast picking,
chords, other strings/performers remain unvalidated. No commit or push requested.

## Previous checkpoint — diagnostic plotted, settings unchanged (2026-11-02)

User requested plotting the suspicious window while explicitly avoiding
overfitting to an outlier. Added section 12 to notebook 04 and matching script:
side-by-side original waveform, full search-lag correlation curve, and first
30 ms selected-shift overlay for window midpoints 1.01 and 3.81 s. Original
waveform panels share amplitude limits; overlay panels have separate scales.
No FFT, smoothing, preprocessing, thresholds, or estimator changes.

P2 results: 1.01 s ([46080:50880]) selects lag 586 / 81.91 Hz / 0.8810;
3.81 s ([180480:185280]) selects lag 151 / 317.88 Hz / 0.8498. The former
has a distinct correlation peak. The latter has tiny local bumps on a broadly
declining correlation curve, not the same clear cycle structure. The estimator
requires a positive local maximum but not prominence above neighboring shifts.
A high score can reflect slowly varying structure; physical origin unknown.

Additional measured observation: suspicious-window raw RMS 0.00218877, mean
-0.00212395, centered RMS 0.000528726. Raw RMS includes the mean whereas the
pitch estimator subtracts it. Documented this difference for later investigation;
do not change preprocessing from this single example. Clear-window raw RMS
0.00927535 versus centered RMS 0.00918610. No listening test performed by agent.

All 12 notebook code cells executed directly with project Python, schema valid,
plot visually inspected. Diagnostic lag/score matches shared estimator via
assertions. Existing P2 CSVs remain byte-for-byte unchanged, 528/591 passing
windows retained. Saved plot: outputs/pitch_tracking/
P2_singlenotes_candidate_diagnostic.png (ignored by Git). No new dependencies.

Next: user reviews section 12, then expand to 0–30 s with settings fixed and
inspect more note interiors/transitions. Do not add a 318 Hz exception or tune
to this timestamp. This is development data, not held-out evaluation. Changes
remain uncommitted; no push requested.

## Previous checkpoint — review and listening clarification (2026-11-02)

User returned to understand the current work, limitations, and next steps before
further coding. Reviewed the estimator, window loop, MIDI loader, background
selection, and acceptance loop. No algorithm changes made in this review.
All 15 repository tests pass. Notebook/script sources match. Re-executing the
decision cell against saved raw values reproduces P1 546/591 and P2 528/591.
Additional review checks cover new-pitch reacquisition, NaN breaks, low-level
periodic candidates, and stable misleading candidates. These filter checks are
ad hoc; the repository test suite does not yet include the acceptance loop.

Important listening clarification: the user listened near P2 3.8 s and heard
nothing unusual. Do NOT equate the suspicious 317.88 Hz estimate with an audible
new note or a proven physical source. Reading original samples [180480:185280]
(3.76–3.86 s) reproduces lag 151, frequency 317.8808 Hz, similarity 0.84978,
level ratio 8.5336, and zero cents difference from the prior candidate. It passes
the implemented checks. Its acoustic cause has not been established. Historical
wording below calling this a wrong pitch/failure is stronger than the evidence;
describe it as a suspicious estimate inconsistent with surrounding played notes.

Review concerns: (1) the manual reference uses only 2 overlapping P1 / 4 P2
windows and isn't confirmed pure noise; changing to a later excerpt excluding
that reference makes section 7 raise an error; (2) the continuity check compares
adjacent raw candidates, not an established note, and 80% sample overlap means
briefly stable incorrect estimates can pass; (3) the fixed similarity cutoff
inherits the estimator's lag/window-length bias; (4) acceptance logic remains
duplicated in notebook/script and lacks committed regression tests. Current
function separation and exact original sample indices are sound for this lesson.

Recommended next step: one short diagnostic of the 3.76–3.86 s window versus a
clear ringing window, showing the waveform, lag curve and selected shift. Keep
thresholds fixed and don't infer a physical cause from an algorithmic repeat.
Then expand the excerpt (e.g. 0–30 s, retaining the reference) to inspect more
notes before choosing changes. Later extract the acceptance calculation into a
small tested pure function when reusing it; don't perform a broad refactor now.
After broader inspection, group passing windows into note events and handle
repeated picking with attack detection. Chords and string/fret assignment remain
future work. Latest changes are still uncommitted on main; no push requested.

## Previous milestone — first pitch acceptance experiment (2026-10-29)

User authorized coding level/repetition/continuity checks. Lesson 04 sections
9–11 now add an explicit, readable loop AFTER the unchanged raw measurements:
RMS/reference >= 3, similarity >= 0.80, and current/previous windows both pass
those checks with candidate frequencies within 50 cents. Cents = absolute
1200*log2(current/previous). These are unoptimized trial parameters, fixed for
P1/P2. MIDI is not used in decisions. Reference intervals are still manual and
provisional; no automatic noise estimation or note-end detector exists.

The previous RAW eligible window is used, not the last accepted pitch. New
pitches can qualify on their second eligible window; no backfill. First window
fails. NaN/below-level/weak-repetition breaks the sequence. This adds one hop
(20 ms) for confirmation, beyond a 100 ms window; overlap is not independent
confirmation. Bends/fast notes/attacks may be rejected; stable wrong tones pass.

Outputs retain raw Hz and exact original sample boundaries. Separate
*_pitch_decisions.csv stores each Boolean check, cents change, accepted Hz (NaN
if rejected), all applicable rejection reasons, background bounds/reference RMS,
and settings. *_pitch_filter.png shows full and zoomed raw/passing candidates,
level and similarity. Green means passes trial rule, not proven correct pitch.

Results: P1 546/591 and P2 528/591 pass. These are counts, NOT accuracy. All
complete windows in five selected ringing intervals pass for both. P2 607.59 Hz
at 7.71 s fails level; 666.67 Hz at 3.87 s fails cents agreement. BUT 317.88 Hz
at 3.81 s passes all checks, a known remaining mismatch. Do not retune solely
for it. The rule checks adjacent agreement, not tracking a whole fading note.
P1 retains two windows in the second candidate tail; it is not confirmed noise.

Next with user: read section 9, trace one accepted/rejected row in section 10,
and inspect the known 3.81 s failure by changing zoom to 3.5–4.2 s. Then inspect
more audio with settings fixed before note-event grouping/attack detection.
Keep the pace educational; avoid repeatedly tuning the same two excerpts.

Verified synthetic stable tones, octave change/reacquisition, low-level periodic
signals, loud weak-repetition signals, missing pitches, and the known stable-
wrong-tone limitation. Ran both players and checked raw measurements/sample
bounds unchanged, accepted Hz unchanged, rejected Hz NaN, and reasons present
for every rejection. Notebook cells executed directly with project Python, not
live Jupyter. No new dependencies; no commit/push requested.

### Previous step — background comparison

Latest step: user agreed to inspect background-relative level before choosing
a filter. Sections 7–8 in lesson 04 now reuse existing 100 ms RMS windows,
selecting only windows fully inside an editable candidate interval by ORIGINAL
sample bounds. Median selected RMS is a provisional reference; each window's
RMS divided by it gives a level ratio. It is not SNR or an acceptance threshold.
The pitch estimator and raw CSV measurements remain unchanged.

P2 candidate 7.70–7.86 s contains four overlapping windows, median RMS
0.0002565. Selected ringing intervals span approximately 21–52 times that
reference. A second candidate tail at 11.60–11.90 s has median ratio 1.86,
range 1.54–2.29. These are visually selected low-level intervals, NOT confirmed
background-only recordings; no listening verification was performed. They may
contain residual vibration or handling noise. MIDI gaps do not define silence.
P1 has its own editable 7.78–7.90 s selection with only two complete windows.
This is a small teaching comparison, not a robust automatic noise estimator.

Previous next-step plan (now implemented above): try one explicit level-plus-
similarity rejection experiment on both players. Preserve raw candidates and
rejection reasons; rejected means "no reliable pitch", not necessarily silence.
Similarity cannot guarantee correct pitch, and thresholds may cut quiet tails.
At that checkpoint persistence was still only an idea; avoid delaying/missing
fast notes by adding it without evidence. Do not assume the first 0.5 s is quiet.

Executed all eight notebook code cells directly in the project interpreter,
validated notebook format, and visually checked the P2 comparison plot. Also
ran the script for P1 and checked that existing numerical estimates were
unchanged, selected windows lie wholly inside the interval, and the median
reference ratio equals one. No new dependencies or algorithm tests were needed.
Generated comparison PNGs are ignored under outputs/pitch_tracking. Changes
remain local and uncommitted; no push was requested for this step.

The user resumed on 2026-10-29 and explicitly asked to test P2. The unchanged
raw baseline has now been tested on P2 before choosing rejection thresholds.
At that transfer checkpoint, no rejection rule was implemented. See the transfer findings
below; they qualify the earlier plan to use simple RMS/similarity thresholds.

Earlier P1 verdict: lesson 04's 100 ms / 20 ms sliding baseline follows the three
ringing pitches consistently. Median estimates over 0.5–3.5, 4.5–7.5, and
8.5–11.5 seconds are ~82.76, 87.91, 93.02 Hz versus nominal MIDI references
82.41, 87.31, 92.50 Hz. Low-level gaps/tails can still yield spurious pitches;
the final frame is ~888.89 Hz. This is a promising baseline, not a validated
accuracy claim. Actual tuning and estimator error have not been separated.

Learning checkpoint: user understands harmonics and that similarity is
autocorrelation, not label accuracy. Clarify again if needed: autocorrelation
compares samples WITHIN ONE analysis window against time-shifted copies of that
same window. It does not compare a window against the previous sliding window
or against MIDI. A lower score means weaker repeating-pattern agreement under
this calculation; gain reduction alone does not necessarily reduce normalized
similarity. Noise, changing vibration, and the shorter-window overlap penalty
can affect it. The 100 ms scores cannot directly inherit half-second thresholds.

Earlier next-step direction (filter experiment now implemented above):
1. Review P2's high-score mismatches: a fixed lower bound on RMS and similarity
   alone cannot cleanly separate all observed ringing and gap candidates.
2. Design a small explainable rejection experiment that considers nearby pitch
   consistency/local level changes as well. Preserve raw candidates and test both
   P1 and P2 without narrowing pitch bounds to their known labels. Timing delay
   and rejection of valid fast/leaping notes must be considered. No rule had been selected at that point.
3. Later group accepted pitch windows into note events; repeated same-pitch
   picking will need attack detection. No extended MIDI alignment project first.

Keep explanations focused on key math, code, architecture, and concrete concerns.
Preserve the user's preference against overengineering and black-box additions.
All 15 tests passed at wrap-up. Lesson 02 now contains the user's saved 5–5.5 s
experiment; preserve that edit (the older script still defaults to 3–3.5 s).
Read the Git checkpoint below before assuming changes reached GitHub.

### P2 transfer results — 2026-10-29

`04_tracking_pitch.ipynb` and its script now have an explicit `player = "P2"`
setting. Change to P1 to reproduce the previous run. Both use 0–11.9 s, 48 kHz,
100 ms / 20 ms windows, 60–1,000 Hz bounds, and the unchanged `pitch_analysis.py`.
P2 stereo channels are exactly identical in this excerpt; use the left channel
as already defined. No mixing, normalization, gating, or MIDI timing changes.

P1 and P2 have the same 591 original sample windows. P2 has finite mathematical
candidates in all 591, not 591 reliable notes. Median candidate frequencies in
the previously chosen 0.5–3.5 / 4.5–7.5 / 8.5–11.5 s inspection intervals:
- P1: 82.76 / 87.91 / 93.02 Hz.
- P2: 81.91 / 86.96 / 91.95 Hz.
- Nominal MIDI references: E2 82.41 / F2 87.31 / F#2 92.50 Hz.
These are selected-interval summaries, not held-out accuracy measurements.
P2 follows the ringing sequence but produces larger excursions in gaps/tails.

Key counterexample: P2 midpoint 7.71 s yields 607.59 Hz with similarity 0.9013
and RMS 0.0005629, in a gap between labels. Another at 3.87 s yields 666.67 Hz,
similarity 0.8780, RMS 0.001389. Some P1 ringing windows have both lower RMS and
lower similarity than that second mismatch. Thus simply raising both minimum
thresholds cannot retain every useful P1 frame while rejecting all P2 mismatches.
This demonstrates periodicity is not correctness. Physical causes of the spurious
matches have not been identified; do not call every gap pure noise or silence.

For inspection only, candidates outside 75–110 Hz numbered 9 for P1 and 35 for
P2. This broad band is specific to these three notes, NOT a filter, validity label,
or general error metric. Full P2 finite range: 60.15–923.08 Hz. Last frame:
271.19 Hz, similarity 0.3515, RMS 0.000588. All finite candidates stay plotted.

Regenerated P1 with the same current code and confirmed its saved numerical
results are unchanged. Verified P1/P2 window indices match, 591 frame widths/
hops/midpoints/RMS, and notebook schema. Executed notebook cells directly with
project Python (not live kernel), inspected plot, and added results explanation
to notebook 04. Separate CSV/PNG outputs remain in ignored outputs/pitch_tracking.
No algorithm changed; existing unit tests were not unnecessarily rerun.

P2 is now an inspected development example. If used to choose thresholds, it
must not later be presented as an untouched test set. Only its first three notes
were analyzed; this does not establish full-dataset, chord, or fast-note performance.

## Purpose

The owner is a Statistics / Data Science student learning Python, software
engineering, signal processing, statistics, and eventually machine learning
through personally recorded guitar audio. Understanding the work matters more
than quickly producing a large application.

The user clarified the primary end goal on 2026-10-27: automatically transcribe
guitar recordings into tablature, ultimately including faster picking, overlapping
notes, and chords. Pitch, attack timing, and plausible string/fret assignments
must guide the roadmap. Bends, vibrato, slides, performance comparison, and a
possible explanation layer are secondary interests. Full note-event transcription,
multi-pitch detection, and tab generation are not implemented. Preserve the tutoring pace; explain one small
step before implementing it, and avoid prematurely adding models or architecture.

## How to collaborate

- Act as a tutor and pair programmer. Explain concepts, make a small change,
  review the result together, and check understanding before advancing.
- The user has authorized direct edits in VS Code's project and running checks.
  Keep edits small; this is not permission to build the entire roadmap at once.
- Explain math with intuition, a formula, a small example, and its Python mapping.
- Use concise, natural comments about reasoning and assumptions. Avoid comments
  that merely narrate obvious code, excessive docstrings, or elaborate architecture.
- Explain errors before fixing them and explain unfamiliar terminal commands.
  Give a few steps at a time, not a long command sequence.
- Teach Git through meaningful working milestones. Review staged files before
  committing; do not include environments or GarageBand editing projects.
- Call out statistical concerns when they arise, such as inconsistent measurement
  windows, recording gain, repeated takes, and later data leakage.
- Keep this handoff updated at meaningful milestones, including unresolved issues.
- User reiterated on 2026-10-28: avoid overengineering. Prioritize understanding
  the data and building a useful note-analysis program. Do not turn the small
  MIDI timing mismatch into a prolonged alignment project before progressing.
- With each change, explain the key math/code and system architecture: inputs,
  outputs, module responsibilities, and why the change is useful. Raise concrete
  concerns revealed by the data or design; keep the next learning step small.

## Environment and files

- Project: `/Users/wesley/Documents/Coding-Projects/01-Active/Guitar_ML`.
- GitHub: https://github.com/wesley-hawes/Guitar_ML ; branch: `main`.
- macOS, VS Code, Python 3.13, local `.venv`; Conda `(base)` may also appear.
- Activate with `source .venv/bin/activate` from the project root.
- Install with `python -m pip install -r requirements.txt`.
- Run comparisons with `python compare_recordings.py` from the project root.
  Close both plots to finish. `python main.py` still runs the earlier single-file demo.
- The user sometimes runs bare `cd`, which returns to the home directory.
  Explain working directories when a file cannot be found; don't recreate files.
- `main.py`: reusable audio loading/RMS functions and the earlier single-file demo,
  now guarded by `if __name__ == "__main__"` so imports have no plotting side effects.
- `compare_recordings.py`: onset-aligned batch amplitude comparisons.
- `test_compare_recordings.py`: seven focused unittest checks.
- `requirements.txt`: NumPy, SoundFile, Matplotlib (currently unpinned), Mido >=1.3,<2.
- `.gitignore`: excludes `.venv/`, `__pycache__/`, `.DS_Store`, and `*.band/`.
- `README.md`: current setup, behavior, and limitations.

## Latest working checkpoint — sliding-window analysis (2026-10-28)

This checkpoint supersedes earlier suggestions to focus next on MIDI timing
correction. User wants practical data understanding and a continuous analysis
prototype, with learning explanations and concrete concerns at each change.

Implemented `04_tracking_pitch.ipynb` and matching `pitch_tracking_lesson.py`:
1. Load P1 direct-input audio at 0–11.9 s (48 kHz mono), original samples 0:571200.
2. Define 100 ms windows (4,800 samples), 20 ms hops (960 samples), 60–1,000 Hz bounds.
3. Explicitly loop over all 591 complete windows, computing candidate pitch,
   original-window RMS, lag, and similarity. First midpoint 0.05 s, last 11.85 s.
4. Load MIDI labels ONLY AFTER all audio measurements; use as unchanged plot overlay.
5. Plot all candidates, RMS in dBFS, and similarity on a common time axis.
6. Save CSV/PNG in ignored `outputs/pitch_tracking/`, prefixed by dataset folder name.

Architecture: `pitch_analysis.py` is the shared single-window calculation with
no MIDI/files/plots. Input is one-dimensional samples, rate, and frequency bounds;
output is candidate Hz, lag in samples, and normalized similarity. Notebook/script
owns loading, sample-index window traversal, diagnostics, plotting, and saving.
`guitartechs_midi.py` is the separate annotation reader, used only for comparison.
Lesson 03 now imports this same pitch function rather than duplicating it; its
three default results are unchanged. Lesson 02 retains its explicit teaching
calculation and the user's edits; it was not overwritten.

The estimator returns NaN for a constant window or no positive local peak;
invalid/too-short input raises ValueError. No RMS/similarity filtering, smoothing,
MIDI-based selection/correction, label time shifts, pitch-name assignment,
note grouping, onset detector, chord decoder, training, or tabs were added.

Observed results: 590 of 591 windows have mathematical candidates (not 590
reliable notes). Candidate pitch follows E2/F2/F#2 in ringing portions. Medians
at 0.5–1.0, 4.5–5.0, 8.5–9.0 s are ~82.62/87.91/93.02 Hz. Low-level gaps/tails
produce large excursions: overall finite range 60–888.89 Hz; final frame is
888.89 Hz at RMS ~0.000115 and similarity ~0.308. All excursions stay visible.

Learning points / concerns:
- Overlapping windows are observations, not separate note events or independent
  test examples. A 20 ms hop does not imply 20 ms onset accuracy; timestamps are
  midpoints of 100 ms spans. Original start/exclusive-stop samples are in the CSV.
- Similarity falls from ~0.97 in half-second windows to ~0.88 in 100 ms ringing
  windows partly because the score sums N-k overlapping pairs but divides by
  zero-lag energy over N samples. Do not reuse thresholds across window sizes.
- RMS plot uses 20*log10(RMS), flooring only the display at 1e-12 (-240 dBFS).
  Original RMS is preserved in CSV; no audio normalization or noise gate is applied.
- Label gaps can contain ringing; the MIDI is neither perfectly timed nor perfect
  ground truth. Keep the mismatch visible rather than tuning to exact timestamps.
- 60–1,000 Hz does not cover all high notes in the full dataset. This is a
  single-note development example, not evaluated chord/full-dataset performance.

Validation: known tones with stronger second harmonics at 44.1/48 kHz passed
within 1% using 100 ms windows; gain/DC invariance, constant-input NaNs, and
invalid input checks passed (`test_pitch_analysis.py`, four test methods).
Executed all six lesson cells, validated notebook structure, inspected the plot,
and verified 591 windows, 4,800-sample widths, 960-sample hops, midpoints, and RMS.
Re-executed lesson 03 to verify unchanged results after sharing the estimator.
Saved outputs were generated by executing cells with project Python, not via a
live Jupyter kernel. No packages added in this step; Mido was already installed.

Next learning step: user reads lesson 04's explicit sliding loop and three-panel
plot. Discuss the failure regions, then try a small explainable no-reliable-pitch
rule and check another excerpt/performer before tuning. Keep progress aimed at
note events and tabs; do not start an extended timing audit or large ML system.

## Current direction — audio to tablature (2026-10-27)

### Guitar-TECHS downloads inspected — 2026-10-28

User downloaded P1/P2 chords, scales, singlenotes, techniques and P3_music into
`/Users/wesley/Downloads`. All nine distinct ZIPs match the official Zenodo MD5
checksums: https://zenodo.org/records/14963133. `P3_music (1).zip` is an identical
duplicate; nothing was deleted, moved, or extracted during inspection.
P1_singlenotes and P2_singlenotes were already extracted in Downloads.

Labels ARE present under each archive's `midi/` directory as `.mid` files.
Each has corresponding `audio/directinput/directinput_<name>.wav` and
`audio/micamp/micamp_<name>.wav`; ego/exo audio is MP3 under `video/`.
Excluding macOS metadata, each P1/P2 group contains 28 chord MIDI files,
12 scales, 1 single-note sequence, and 5 techniques. P3 has 12 music MIDI files:
104 label files total, each paired with both WAV capture types. Ignore __MACOSX
and ._ resource-fork entries when constructing manifests.

Single-note example pair:
`/Users/wesley/Downloads/P1_singlenotes/audio/directinput/directinput_allsinglenotes.wav`
and `/Users/wesley/Downloads/P1_singlenotes/midi/midi_allsinglenotes.mid`.
Both players' WAVs are 552 seconds at 48 kHz: long note sequences, not isolated
WAVs per note. P1 is mono, P2 stereo; DI is PCM24 and amp-mic PCM16.
Both MIDIs are format 1, 960 ticks/quarter, with tempo 1,000,000 microseconds per
quarter (60 BPM), a metadata track and six tracks named e/B/G/D/A/E.
Note-on/off events and string-track identities were inspected directly; no MIDI
package installed. P1 uses channel 0 across all string tracks, so channel alone
must not be used as a string label. P2 uses separate channels as well as tracks.
P1 contains 142 positive-velocity note-ons; P2 contains 137. Some P1 low-velocity
events occur outside the expected ascending string sequence, and P2 D has 22
events. These deserve review, not automatic deletion or assumptions of perfect
ground truth. Pitch range includes MIDI 86 (~1175 Hz), exceeding the lesson's
current 1,000 Hz upper bound.

Official dataset notes warn of up to 100 ms signal misalignment. Verify audio
and MIDI timing on a short excerpt before scoring attacks or training. The
official site https://guitar-techs.github.io/ states CC BY 4.0 and requests citation.
Next small step: read a short P1 direct-input/MIDI pair, display note labels as
a table and overlay timings on audio before building a training dataset. Keep
all captures of one performance together in future train/validation/test splits.
No full training, automated note-label alignment, or audio quality audit performed.

### First Guitar-TECHS label overlay — 2026-10-28

User authorized taking a short single-note excerpt and comparing labels against
waveform plots. Added `03_guitartechs_labels.ipynb` and its matching
`guitartechs_labels_lesson.py`. Five code cells read P1 direct input 0–11.9 s
(48 kHz mono, samples 0:571200), load MIDI labels, plot unchanged label spans
and a zoom around F2's start, estimate pitch in one half-second segment per label,
and plot nominal label frequencies separately from predicted frequencies.

User installed Mido successfully in `.venv` after tool-environment installation
failed. Added `mido>=1.3,<2` to requirements. `guitartechs_midi.py` uses Mido to
preserve track identity, order events, convert ticks with the tempo map, and
pair note-on/off messages (including velocity-zero note-ons). It reports invalid
pairings instead of silently dropping them. No custom binary MIDI decoder remains.

First supplied labels (all low-E string, standard-tuning frets 0/1/2):
- E2 (MIDI 40): 0–3.7354167 s; target 82.4069 Hz.
- F2 (MIDI 41): 4.0208333–7.8 s; target 87.3071 Hz.
- F#2 (MIDI 42): 8.0260417–11.790625 s; target 92.4986 Hz.
Sampling starts 0.5 s after each supplied onset, for 0.5 s. Original sample
windows: [24000:48000], [217000:241000], [409250:433250]. Same 60–1,000 Hz
autocorrelation rule returns lags 581/547/516, frequencies 82.6162/87.7514/93.0233
Hz, similarities ~0.971/0.970/0.974. Expected label pitch does not enter estimator.
These nominal-versus-measured differences are not established tuning errors;
integer lag, actual tuning, and changing pitch can contribute. No threshold tuned.

The F2 waveform attack visibly precedes the MIDI marker by a few tens of ms.
No automatic alignment correction was made. MIDI note-off is not guaranteed to
mark the final audible decay. First note has no preceding context (start at zero).
This is label-guided window selection, not automatic event detection, a
sliding-window implementation, independent validation, or model training.

All five notebook cells executed with the project interpreter; outputs saved,
notebook schema validated, both plots visually inspected (not a live kernel test).
Four MIDI tests in `test_guitartechs_midi.py` pass: tempo changes/zero-velocity note-offs,
same channel/pitch on different tracks, invalid pairing errors, and asynchronous
file rejection. Six known-tone/stronger-harmonic pitch checks at 44.1/48 kHz pass;
excerpt labels and original sample bounds also verified. Original WAV/MIDI and
the user's modified frequency notebook were left untouched.

Next teaching checkpoint: read notebook 03's first three cells and inspect
what the shaded labels mean; then review the prediction table/plot in cells 4–5.
Keep timing mismatch visible before proposing any alignment rule or training.

### Transcription roadmap

This section supersedes the earlier suggestions to keep tuning sustained-note
end thresholds. The user explicitly asked whether that work is overengineering
relative to automatic tabs for faster picking and chords. It is time to move on.
The RMS lessons remain useful foundations, not the final transcription algorithm.

- User's latest listening observation with headphones: low E take02 seems to
  fade into background around 14 seconds. Earlier estimates were about 11 and
  15 seconds; treat these as approximate subjective observations, not exact
  endpoint labels. Measured RMS approaches pre-note background near 15 seconds.
  RMS alone cannot establish how much remaining energy is guitar versus noise.
- Do not keep adjusting thresholds to resolve a 14-versus-15-second endpoint.
- A rule that waits for quiet between notes will miss new attacks during ringing.
  The current 100 ms confirmation rule is only an isolated-note learning baseline,
  not established as suitable for fast picking or polyphonic playing.
- The spectrum and first autocorrelation candidate-fundamental lesson are now
  implemented (see below). Review the similarity loop and peaks before advancing.
- First transcription milestone: recover the ordered pitches and approximate
  attack times of a short, known single-note phrase, then map them to tentative
  string/fret positions under stated tuning and position constraints.
- Same pitch can occur on multiple strings: pitch detection alone does not
  uniquely identify the player's original fingering. Distinguish plausible tabs
  from recovered actual string/fret assignments.
- Faster passages and overlapping notes/chords require later onset and multipitch
  work; multiple spectral peaks can also be harmonics of a single note. Compare
  against known recorded phrases, and keep fresh takes for validation.
- Later, consider evaluating an existing polyphonic transcription model as a
  baseline (e.g. Spotify Basic Pitch produces note events/MIDI, not guitar tabs).
  No model was installed, selected for production, or used on the recordings.

## First frequency spectrum — 2026-10-27

User authorized starting frequency detection and committing/pushing to GitHub.
Added `02_understanding_frequency.ipynb` and matching `frequency_lesson.py`,
using existing dependencies. Four visible code cells select low E take02's
3.0–3.5 s interval (samples 132300:154350), plot its location, subtract its mean,
apply a Hann window, compute NumPy rFFT magnitudes and frequency labels, then
plot a spectrum. Sample rate is read from the file, not hardcoded. Selection is
manual and editable, not a new silence assumption or automatic onset rule.

22,050 samples at 44,100 Hz give a 0.5 s segment and 2 Hz FFT bin spacing;
spacing is not a guarantee of pitch accuracy. Largest nonzero bin is 164 Hz
for low E take02; the lower peak near 82 Hz and higher harmonics illustrate an
octave-error risk. The same interval in low E take01 peaks at 82 Hz, A takes
at 110 Hz, and C takes at 262 Hz. These are dominant bins, not automatically
identified fundamentals or verified note labels. Filename labels do not enter
the peak selection. The graph shows relative magnitudes and cannot compare
recording volume. Full nonzero spectrum is searched; plot zoom is 0–1,000 Hz.

Executed the four cells with the project interpreter and saved example outputs;
validated notebook structure and visually inspected the spectrum. This was not
a live Jupyter kernel test. Exercised the actual FFT cell with known 110 Hz tones
plus DC and with a stronger 220 Hz harmonic, at both 44.1 and 48 kHz; expected
dominant bins passed. Exact silence raises an explanatory error. Noise can still
produce a largest peak: no note-presence detector is claimed. Seven existing
comparison tests passed. Audio and batch detection code were not changed.

The user subsequently understood the difference between frequency positions and
harmonic strengths, and the shifted-waveform explanation of autocorrelation.
They explicitly requested coding it; the following milestone advances this lesson.
Avoid using a 0.5 s frame as a universal fast-note setting. No multipitch model,
note-name mapping, or tablature generator added.

### Autocorrelation addition — 2026-10-27

Added sections 5–6 (two code cells) to the same notebook and matching script.
Use the same mean-centered 3.0–3.5 s segment, without the FFT Hann taper. An
explicit loop multiplies overlapping sample pairs at each lag and sums them,
dividing by the zero-lag squared sum. No FFT shortcut or new dependencies.
Longer shifts contain fewer pairs, causing a mild shorter-lag preference;
this is documented, not presented as overlap-corrected correlation or confidence.

Search range is fixed at 60–1,000 Hz, converted with ceil(sr / max_frequency)
and floor(sr / min_frequency), i.e. 45–735 samples at 44.1 kHz. Zero lag is not
a candidate. Choose the highest positive local maximum within the bounds;
compute one extra lag to check the upper boundary's neighbor. Report competing
peaks, best lag, period, candidate frequency, score, and prior FFT strongest bin.
Plot similarity versus lag plus an amplitude-preserving overlay of three periods.
No filename-based expected pitch, per-take parameter tuning, interpolation,
note-presence threshold, multiple-note tracking, or chord decoder is used.

Unchanged-settings development results on all six current takes:
- Low E take01 and take02: lag 538, 81.97 Hz (scores 0.975 / 0.979).
- A take01: lag 399, 110.53 Hz; A take02: lag 400, 110.25 Hz (scores ~0.981).
- C take01 and take02: lag 168, 262.50 Hz (scores 0.992 / 0.989).
For E take02 the competing lag 270 (163.33 Hz) scores only 0.114, illustrating
why its stronger second harmonic does not imply a fundamental of 164 Hz.
The scores are similarity measures, not accuracy probabilities. Integer-lag
rounding affects pitch precision; multiple-period/harmonic peaks remain a risk.

Executed all six notebook code cells and saved outputs, validated notebook
structure, and visually inspected the new plots. Not a live Jupyter kernel test.
Exercised the actual autocorrelation cell on 12 synthetic cases: 82.41, 110,
261.63 Hz at 44.1/48 kHz, with and without a second harmonic twice the fundamental
amplitude. All estimates were within 0.5% of the known frequency. Gain scaling
plus a DC offset preserved each chosen lag. Silence, constant, and too-short
inputs raised errors. Previous FFT synthetic checks also passed. These checks
do not establish real-world performance, especially for noise, chords, or attacks.

Next tutoring checkpoint: user reads section 5's slice/product loop and section
6's plots, understands score normalization and candidate selection. Then examine
stability across ringing intervals and move toward a known melody with shorter
sliding windows. Keep new takes for evaluation instead of tuning to these six.

## Current comparison milestone — 2026-10-25

The user explicitly requested a loop over WAVs with granular amplitude analysis
and comparisons anchored to each note's onset sample. This is now implemented
in `compare_recordings.py`; the single-file exercise remains in `main.py`.

- Loops through `data/raw/*_note_dry_take*.wav`; currently six files (A, C, low E,
  two takes each). Excludes the older processed/unknown-conditions `Test_1_Cmaj.wav`.
- Uses the left channel (or mono), without per-file normalization. Requires finite
  audio and a full 8 seconds after onset; unusable files are skipped with a reason.
- Estimates background RMS from the first 0.5 seconds, which must be quiet.
  Finds three consecutive 10 ms frames above 6 times that RMS (floor 1e-5),
  then selects a nearby sample threshold crossing. This is a heuristic, not a
  validated or exact physical onset. Brief clicks are rejected, but sustained
  handling noise or soft attacks can still fool it. Only the first event is used.
- `ONSET_OVERRIDES` accepts manually reviewed original-file sample indices.
- Measures consecutive 50 ms windows from the onset over the same 8-second span.
  At 44,100 Hz: 2,205 samples/window, 160 windows/file. Partial windows are
  dropped, never zero-padded. Relative time is `(sample_index - onset) / sr`.
- Plots RMS and a 0.10–0.15 second post-onset waveform close-up, grouping takes
  by note with shared amplitude scales. Alignment is by onset, not waveform phase.
- Saves `comparisons.png`, `onset_review.png`, and `measurements.csv` to
  `outputs/amplitude_comparison/` (ignored generated output). CSV includes onset
  method, background threshold, original start/exclusive-stop indices, relative
  window midpoints, RMS, and absolute peaks. No pitch/FFT/ML was added.
- Detected onset samples in the current files: A take01 89373, A take02 88489;
  C take01 68649, C take02 89384; low E take01 68712, low E take02 108551.
  Visually reviewed the onset plots; estimates include early attack transients
  and should not be interpreted as precise starts of periodic string motion.
- Seven unittest checks passed, covering silence-offset invariance, isolated
  clicks, noise-only input, gain scaling, two sample rates, partial/short windows,
  and mono/manual overrides. Real-file checks verified 960 CSV windows, original
  sample boundaries, equal comparison spans, and no data cropped by plot limits.
- The batch milestone originally had no notebook. A later guided quiet-region
  lesson is now available; see its setup/status below.

## Guided quiet-region lesson and notebook setup — 2026-10-25

- Created `01_understanding_quiet.ipynb` and matching `quiet_region_lesson.py`.
  Six short code cells cover loading, waveform inspection, selecting an editable
  quiet interval, one RMS calculation, looping over equal windows, and plotting
  a candidate threshold. The batch detector is unchanged.
- Uses C take01 with candidate quiet interval 0.20–1.00 seconds (samples
  8820:44100); this is a proposed human-reviewed region, not automatic silence
  detection or a fixed requirement for all files. There are 16 complete 50 ms
  windows. Quiet median RMS is about 0.00015758, maximum about 0.00016201.
- Candidate threshold is 1.5 times that maximum, about 0.00024301. This is an
  explicit teaching heuristic, not a validated threshold. Being above the same
  windows used to choose it is not independent validation. Start/end decisions
  and sustained-threshold logic are deliberately not implemented in the lesson.
- Executed all six code cells in sequence using the project interpreter and
  saved example outputs in the notebook; validated structure using Anaconda's
  nbformat. This was NOT a live Jupyter-kernel test. Inspected final plot.
- The user asked to install notebook support. Project `.venv` lacks ipykernel;
  VS Code lacks Microsoft's Jupyter extension. Anaconda base already has Jupyter.
  Pip and extension install attempts failed due DNS/network resolution from the
  tool session (pypi.org and marketplace.visualstudio.com). No packages or
  extensions were installed, and no kernel was registered.
- Added `requirements-notebook.txt` and README setup commands. User should run
  installation in their own terminal, install `ms-toolsai.jupyter`, and select
  `Guitar ML (.venv)` after registering it. The lesson script already works with
  existing dependencies if notebook setup is still unavailable.
- Next teaching step: open the notebook, select the project kernel once setup
  succeeds, run only the first two cells, and discuss the waveform before
  continuing. Then review the candidate quiet region and one RMS calculation.

### Additional quiet-region check — A take01

The user confirmed no playing during `A_note_dry_take01.wav` from 0 to 1.2
seconds. Added section 7 to `01_understanding_quiet.ipynb` and the matching
Python lesson. Kept the C-derived threshold fixed at approximately 0.00024301;
did not refit it to A. Used A samples 0:52920, split into 24 complete 50 ms
windows of 2,205 samples. A background RMS min/median/max were approximately
0.00016459 / 0.00016922 / 0.00026708.

One window exceeded the threshold: samples 46305:48510, 1.05–1.10 seconds.
The other 23 were below. This is a false alarm for a simple per-window activity
rule, since the user identifies this interval as no playing. Its cause is unknown;
do not claim it is a click or note. Saved measured output and graph in the
notebook and validated the cells. No automatic detector was changed.

Next small lesson: inspect the flagged waveform interval, then discuss a
persistence rule (multiple consecutive windows) and its onset-delay tradeoff.
The 24 adjacent windows are not independent recording sessions and do not
establish a general error rate. Keep the threshold fixed while examining this.

### Strong-and-sustained start experiment — notebook section 8

Added two small code cells to the notebook and matching lesson script. Uses the
median of A take01's user-confirmed 0–1.2 second quiet windows as background.
Explicit provisional settings: window RMS > 10 times background for two
consecutive 50 ms windows (100 ms total). The median baseline is about
0.00016922 and trial threshold about 0.00169218. This is separate from the
unchanged C-threshold check in section 7 and does not modify the batch script.

On A take01, the first qualifying pair begins at sample 88200 (2.000 s), with
confirmation available at sample 92610 (2.100 s). Ratios are about 39.90 and
93.74; the highest known-quiet ratio is 1.58. Verified the earliest qualifying
pair and 4,410-sample duration; inspected the saved plot. The reported start is
a 50 ms window boundary, not sample-accurate physical onset. Returning to the
start of that window avoids assigning the later confirmation time as onset.

Keep the 10× / 100 ms settings fixed for new D/G recordings before evaluating
changes. This A example is development data, not held-out validation. Its quiet
excursion is rejected by the strength condition alone, so it does not validate
the persistence condition. Soft/short notes can be missed and sustained noise
can pass. Do not reuse the high start threshold for detecting the note end.

Next teaching checkpoint: user reads section 8 and explains why detection is
confirmed after two windows but the candidate start is the first window boundary.
Then evaluate the same rule on new takes with user-confirmed quiet regions.

### Low E take02 transfer check — 2026-10-26

The user requested the same rule on low E take02, expecting the note around
2.5 seconds and definitely after 2 seconds. Section 8 now has a short input
cell selecting `Low_E_note_dry_take02.wav` and candidate quiet region 0–1.5 s
(samples 0:66150), before that expected attack. Variables in this section are
now generic `test_*` names. Section 7's C-threshold/A-background check is preserved.
The matching lesson script was updated. No batch detector changes.

Settings stayed fixed: 50 ms windows, 10× median background RMS, two consecutive
qualifying windows. The search still covers the whole file; the user's expected
start was not used as a search constraint. Thirty quiet windows yielded median
RMS 0.00016757 and threshold 0.00167566. First qualifying window boundary:
sample 108045, 2.450 s; confirmation boundary: sample 112455, 2.550 s. Ratios
in those two windows were about 51.00 and 68.86. Highest selected-quiet ratio
was 2.70. The result is consistent with the user's rough expectation, but the
window boundary is not an exact physical attack time. This previously viewed
recording is a transfer check, not a wholly untouched validation example.

Executed notebook code cells, saved new section 8 results/plot, verified the
unchanged settings and earliest qualifying pair, and inspected the plot.
Next: review these two windows with the user; then test a new recording under
the same rule before choosing changes or adding note-end detection.

### Low E take02 tail inspection — 2026-10-26

The user thinks the note faded naturally (not confirmed). Numerically inspected
50 ms windows around 15 seconds against the same 0–1.5 second background median
(0.00016757). RMS is about 5.75× background at 14.00–14.05 s, 1.22× at
14.90–14.95 s, and 1.02× at 15.00–15.05 s. From 15.05 to 15.60 s it remains
approximately 0.96–0.99× background and contains nonzero recorded samples.
Last nonzero sample is 688341 (15.608639 s); the following samples are exact
zeros, through the file end at 16.343855 s. Do not count that digital-zero tail
as evidence of physical note decay or treat it as the noise floor.

This supports the user's rough ~15-second return-to-background observation,
without verifying its physical cause or defining an exact endpoint. No new
end detector or threshold has been implemented. Use this as a reviewed example
when discussing a sustained return-to-background rule; avoid tuning the rule
to hit exactly 15 seconds. The start threshold of 10× is unsuitable for the end.

Learning feedback on 2026-10-25: the user finds the comparison script too much
of a black box. Slow down, explain sample rate, indices, RMS windows, and every
chosen constant before adding algorithmic complexity. The user specifically
wants variable leading/trailing silence handled and fair note comparisons.

Current limitations remain: the detector assumes a quiet first 0.5 seconds,
and no note-end detector exists. The 8-second span can include background after
a short note or omit part of a long decay. Passing numerical tests does not
validate musical boundaries or make these recordings a controlled benchmark.

Next design/learning step (not yet implemented): inspect one recording, identify
a representative quiet region and the note's start/end as original sample indices,
and distinguish natural decay into noise from a mute or file cutoff. Use reviewed
boundaries as reference cases before changing automatic start/end detection.
Quiet-region selection must not assume a fixed 0.5-second prefix; if a reliable
noise-only region is unavailable, surface that uncertainty. Compare equal-duration
windows at equal times after onset, while reporting duration separately and
avoiding treating missing/truncated audio as measured silence.

The first C take's export volume changed during setup; comparisons across notes
are exploratory and do not isolate pitch from picking strength or processing.

## Current recording and behavior

Current recording setup, confirmed by the user on 2026-10-24:

- Gain knob: 9 o'clock.
- Guitar pickup: neck pickup.

These settings describe the current recording session; they have not been
confirmed for the earlier `Test_1_Cmaj.wav` recording.

Latest replacement checked on 2026-10-24 at 19:44: `data/raw/C_note_dry_take01.wav`.

- WAV, 16-bit PCM, 44,100 Hz; shape `(458620, 2)`, about 10.400 seconds.
- Left and right channels are identical; compatible with the current left-channel analysis.
- Full-recording RMS is approximately 0.00459138; absolute peak is 0.05480957
  (about -25.22 dBFS), roughly 6 dB above the preceding export peak of -31.25 dBFS.
  No samples reach absolute amplitude 0.999. This does not rule out upstream distortion.
- About 1.5 seconds of quiet lead-in precedes the note. The 50 ms RMS curve
  shows a clear attack and decay, approaching the background level around 7 seconds.
- Initial 0.5-second RMS is approximately 0.00016060. The strongest 50 ms RMS
  window starts at 1.55 seconds.
- This replacement is usable for RMS/decay and onset exploration, although the
  exported signal level is low. A fixed -1 to 1 plot makes it look nearly flat.
- The prior export under this same filename was 13.372 seconds, had no quiet
  lead-in, and peaked at approximately -0.10 dBFS. Duration and timing changed
  as well as level; do not attribute all differences solely to normalization.
- The user's latest GarageBand screenshot shows Auto Normalize, Master Echo,
  and Master Reverb unchecked. The filename says "dry," but bypass of all
  track/master processing has not been independently confirmed.
- The latest export was checked numerically; earlier exports were also plotted.
  No listening test was performed. `main.py` still
  loads `Test_1_Cmaj.wav`; no analysis code was changed.

Additional recordings checked numerically and visually on 2026-10-24:

- `C_note_dry_take02.wav`: 13.372 seconds, peak about -28.64 dBFS,
  approximately 2.0 seconds of quiet lead-in, followed by a clear attack and decay.
- `Low_E_note_dry_take01.wav`: 14.858 seconds, peak about -27.37 dBFS,
  approximately 1.55 seconds of quiet lead-in and a sustained, fluctuating decay.
  Signal drops abruptly near 13.4 seconds; do not treat that cutoff as natural decay.
- Both are 44,100 Hz, 16-bit PCM stereo WAVs with identical left/right channels,
  finite samples, and no samples at absolute amplitude 0.999 or above. Both are
  usable for the current amplitude-analysis work. Note labels are user-provided,
  not verified by pitch estimation. No listening test was performed.
- `Low_E_note_dry_take02.wav` was subsequently checked numerically: 16.344
  seconds, 44,100 Hz, 16-bit PCM stereo, peak about -30.09 dBFS, approximately
  2.45 seconds of quiet lead-in, a decaying RMS profile, and no samples at
  absolute amplitude 0.999 or above. Usable for current analysis.
- Both open A takes were checked numerically: `A_note_dry_take01.wav` and
  `A_note_dry_take02.wav`. Each is 14.858 seconds, 44,100 Hz, 16-bit PCM stereo,
  with approximately 2.0 seconds of quiet lead-in and a decaying RMS profile.
  Peaks are about -28.10 and -29.36 dBFS respectively; all samples are finite
  and none reach absolute amplitude 0.999. Both are usable for current analysis.
- Next recordings: open D string, `D_note_dry_take01.wav` and
  `D_note_dry_take02.wav`, with unchanged settings. Keep all previous takes.
- When comparing takes, align measurement windows relative to each note's onset;
  lead-in lengths and full file durations differ.

`data/raw/Test_1_Cmaj.wav` contains a single C note, not a C-major chord.
The user recorded/exported it through GarageBand. The exact octave, interface,
gain, and effects settings are not confirmed; do not describe it as verified dry
audio. An earlier GarageBand screenshot showed a "Brit and Clean" amp preset.

Verified file properties:

- WAV, 16-bit PCM, stereo, sample rate 44,100 Hz.
- Loaded array: `(458356, 2)`, dtype `float64` using SoundFile defaults.
- Duration: approximately 10.39356 seconds.
- Full-recording left-channel RMS: approximately 0.12186434.
- The first five rows contain zeros in both channels.
- The displayed waveform has a large onset around 0.85 seconds and a decaying tail.
  A near-full-scale peak alone does not establish clipping.

The earlier `main.py` demonstration assumes stereo input and uses a fixed relative path. It plots only
the left channel and does not save the figures. It now slices 3.00–3.05
seconds (indices 132300:134505), plots those 2,205 samples in a second figure,
and reports window RMS approximately 0.10735958. The earlier 1.00–1.05 second
window (indices 44100:46305) gave RMS approximately 0.39458260.
Both plots use amplitude limits
of -1 to 1, and the overview highlights the selected interval. Window dimensions
and both figures were checked; the close-up was visually inspected.
At that earlier milestone no FFT, pitch tracking, or ML was implemented. The batch comparison tests are
described above; these earlier single-file observations remain for reference.

The script has run successfully. RMS was checked with an alternating
`[0.5, -0.5, 0.5, -0.5]` array (expected 0.5), silence (expected zero), and
doubling the signal (expected twice the RMS). These were development checks,
not committed test files.

## Concepts covered

- WAV data vs a GarageBand `.band` project; renaming is not conversion.
- Samples, sample rate, stereo columns, time-step rows, and floating-point data.
- Duration = row count / sample rate; channels share the same timeline.
- Time array = sample indices / sample rate; selecting `samples[:, 0]`.
- Waveform overview vs rapid oscillations visible in a close-up.
- RMS = square root of mean squared amplitude. Ordinary signed means can cancel.
- Digital amplitude and RMS are not perceived loudness or pitch.
- The user correctly explained that adding zeros reduces RMS: the squared sum
  stays constant while the sample count increases.
- Time-window slicing is implemented and reviewed. The user changed the window
  from 1.00–1.05 to 3.00–3.05 seconds and shared the plot and numerical results.
  The later window has smaller amplitudes, a smoother shape, and roughly similar
  cycle spacing. This is a visual observation, not verified pitch tracking.
- Equal-length windows have the same RMS denominator; the later window's lower
  RMS reflects a smaller sum of squared amplitudes. Whole-recording RMS remains
  unchanged when the selected window changes.

## Git checkpoint and next steps

At the 2026-10-28 wrap-up, main and cached origin/main started at `199664a`
("Autocorrelation notebook for naalyzing pitch frequency across notes"). The user
authorized committing and pushing all current project changes, including their
saved lesson 02 experiment, lessons 03/04, shared modules, tests, and handoff.
Look for the commit "Add Guitar-TECHS labels and sliding-window pitch analysis"
in the actual log; do not rely on a self-referential hash in this document.

The wrap-up fetch again failed with `Could not resolve host: github.com`.
The tool environment cannot currently reach GitHub. A local commit does not
establish upload success; check the push result and fresh remote state. If needed,
the user can run `git push origin main` from their network-enabled terminal.
Dataset ZIPs/audio stay in Downloads and generated tracking outputs are ignored.

The user subsequently resumed on 2026-10-29. At that start, local main and the
cached origin/main matched with a clean tree. The P2 transfer edits are a new
working change; see actual Git status before any further commit/push.
