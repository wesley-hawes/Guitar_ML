import unittest
import numpy as np
from attack_analysis import measure_energy_rise
from event_detection import build_note_events


def group(first,last,note=55,name='G3',rate=48000):
    return dict(first_midpoint_seconds=first,last_midpoint_seconds=last,
                note_number=note,note_name=name,median_hz=196. if note==55 else 164.81,sample_rate=rate)


class EventAssemblyTests(unittest.TestCase):
    def test_quieter_entry_is_integrated_and_end_stays_unknown(self):
        rate=48000;t=np.arange(rate)/rate
        x=np.where(t<.5,.1*np.sin(2*np.pi*196*t),.025*np.sin(2*np.pi*164.81*t))
        events,audit=build_note_events([group(.05,.47),group(.55,.95,52,'E3')],measure_energy_rise(x,rate),.0001)
        self.assertEqual(len(events),2)
        self.assertEqual(events[1]['reason'],'pitch+drop')
        self.assertTrue(.49 <= events[1]['estimated_start_seconds'] <= .54)
        self.assertTrue(all(e['end_seconds'] is None for e in events))
        self.assertEqual(events[1]['estimated_start_sample'],round(events[1]['estimated_start_seconds']*rate))

    def test_rise_and_fall_for_one_entry_are_not_two_events(self):
        rate=48000;t=np.arange(rate)/rate
        envelope=np.where(t<.5,.1,np.where(t<.52,0.,.04))
        x=envelope*np.sin(2*np.pi*196*t)
        events,_=build_note_events([group(.05,.49),group(.57,.95,52,'E3')],measure_energy_rise(x,rate),.0001)
        self.assertEqual(len(events),2)
        self.assertEqual(events[1]['reason'],'pitch+rise+drop')
        self.assertFalse(events[1]['review_repeat'])

    def test_repeat_inside_one_group_creates_reviewable_row(self):
        rate=48000;t=np.arange(rate)/rate
        x=np.where(t<.5,.04,.1)*np.sin(2*np.pi*196*t)
        events,_=build_note_events([group(.05,.95)],measure_energy_rise(x,rate),.0001)
        self.assertEqual(len(events),2)
        self.assertEqual(events[1]['reason'],'pitch+rise_repeat')
        self.assertTrue(events[1]['review_repeat'])
        self.assertEqual(events[0]['note_name'],events[1]['note_name'])

    def test_pitch_only_notes_survive_and_unsupported_rises_do_not_invent_notes(self):
        rate=48000;t=np.arange(rate)/rate
        x=np.where(t<.7,0.,.1)*np.sin(2*np.pi*196*t)
        rows=measure_energy_rise(x,rate)
        events,audit=build_note_events([group(.05,.3)],rows,.0001)
        self.assertEqual(len(events),1)
        self.assertEqual(events[0]['reason'],'pitch_only')
        self.assertEqual(audit['rise_assignments'][0]['assignment'],'unassigned_no_pitch_support')
        self.assertEqual(build_note_events([],rows,.0001)[0],[])

    def test_invalid_order_rate_and_tolerance(self):
        for groups in ([group(.5,.8),group(.4,.6)], [group(.1,.2),group(.3,.4,rate=44100)]):
            with self.assertRaises(ValueError):build_note_events(groups,[],.001)
        with self.assertRaises(ValueError):build_note_events([],[],.001,association_seconds=0)
        with self.assertRaises(ValueError):build_note_events([group(.1,.2)],measure_energy_rise(np.ones(4800),44100),.001)


if __name__=='__main__': unittest.main()
