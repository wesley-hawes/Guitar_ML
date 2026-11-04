import unittest
import numpy as np
from attack_analysis import measure_energy_rise
from quieter_transitions import find_quieter_entries


def groups(rate, changed=True, gap=.08):
    return [dict(note_number=55,note_name='G3',first_midpoint_seconds=.1,
                 last_midpoint_seconds=.47,sample_rate=rate),
            dict(note_number=52 if changed else 55,note_name='E3' if changed else 'G3',
                 first_midpoint_seconds=.47+gap,last_midpoint_seconds=.9,sample_rate=rate)]


class QuieterEntryTests(unittest.TestCase):
    def test_drop_to_quieter_tone_at_two_rates_and_gains(self):
        # Supplied pitch groups isolate the corroboration rule; this is not an
        # end-to-end transcription accuracy test or a simulated physical pick.
        for rate in (44100,48000):
            t=np.arange(rate)/rate
            x=np.where(t<.5,.1*np.sin(2*np.pi*196*t),.025*np.sin(2*np.pi*164.81*t))
            for gain in (1.,.2):
                result=find_quieter_entries(groups(rate),measure_energy_rise(x*gain,rate),.0001*gain)
                self.assertEqual(len(result),1)
                self.assertTrue(.49 <= result[0]['change_midpoint_seconds'] <= .54)
                self.assertEqual(result[0]['to_note'],'E3')
                self.assertEqual(result[0]['stop_sample_exclusive']-result[0]['start_sample'],round(.01*rate))

    def test_decay_does_not_qualify_even_with_pitch_change(self):
        rate=48000;t=np.arange(rate)/rate
        x=.1*np.exp(-t)*np.sin(2*np.pi*196*t)
        self.assertEqual(find_quieter_entries(groups(rate),measure_energy_rise(x,rate),.0001),[])

    def test_stop_same_pitch_and_long_gap_do_not_create_entries(self):
        rate=48000;t=np.arange(rate)/rate
        stop=np.where(t<.5,.1*np.sin(2*np.pi*196*t),0.)
        energy=measure_energy_rise(stop,rate)
        self.assertEqual(find_quieter_entries(groups(rate)[:1],energy,.0001),[])
        self.assertEqual(find_quieter_entries(groups(rate,changed=False),energy,.0001),[])
        self.assertEqual(find_quieter_entries(groups(rate,gap=.4),energy,.0001),[])

    def test_exact_window_gap_and_invalid_reference(self):
        rate=48000;t=np.arange(rate)/rate
        x=np.where(t<.5,.1,.025)*np.sin(2*np.pi*196*t)
        self.assertEqual(len(find_quieter_entries(groups(rate,gap=.1),measure_energy_rise(x,rate),.0001)),1)
        with self.assertRaises(ValueError): find_quieter_entries([],[],0)


if __name__=='__main__': unittest.main()
