import numpy
import itertools

from pax import plugin, dsputils

class F90(plugin.TransformPlugin):
    def startup(self):
        self.dt = self.config['sample_duration'] # Some variable from config
        self.reference_baseline = self.config['digitizer_reference_baseline']

    def transform_event(self, event):
        found_pulse = set()
        for s1 in event.s1s():
            fderivative = s1.sum_waveform[2:] - s1.sum_waveform[:-2]
            sderivative = s1.sum_waveform[2:] + s1.sum_waveform[:-2] - 2 * s1.sum_waveform[1:-1]
            try:
                starttime = [index for index, (slope, curvature) in enumerate(zip(fderivative, sderivative)) if (slope > 0.20 and curvature > 0.20)][0] + 1
                print(starttime)
            except IndexError:
                continue
            f10_int = 0.0
            f20_int = 0.0
            f30_int = 0.0
       	    f40_int = 0.0
            f50_int = 0.0
       	    f60_int = 0.0
            f70_int = 0.0
       	    f80_int = 0.0
            f90_int = 0.0
            f100_int = 0.0
       	    f110_int = 0.0
            f120_int = 0.0
            f130_int = 0.0
            f140_int = 0.0
            f150_int = 0.0
            f160_int = 0.0
            f170_int = 0.0
            f180_int = 0.0
       	    f190_int = 0.0
            f200_int = 0.0
            total_int = 0.0
            s1right = s1.right
            for index, hit in enumerate(s1.hits):
                if(hit['found_in_pulse'] not in found_pulse):
                    pulse = event.pulses[hit['found_in_pulse']]
                    found_pulse.add(hit['found_in_pulse'])
                    time = [t for t in range(pulse.left, pulse.right + 1)]
                    signalpe = (-1.0) * dsputils.adc_to_pe(self.config, pulse.channel) * (self.reference_baseline - pulse.baseline - pulse.raw_data)
                    leftbound, rightbound = self.GetIntegrationBounds(int(s1.center_time / 10.0 - len(s1.sum_waveform) / 2.0) + starttime, s1right, pulse.left, 1)
                    if((leftbound != 0) and (rightbound!= 0)):
                        f10_int += sum(itertools.islice(signalpe, leftbound, min(rightbound, len(signalpe) - 1)))
                    leftbound, rightbound = self.GetIntegrationBounds(int(s1.center_time / 10.0 - len(s1.sum_waveform) / 2.0) + starttime, s1right, pulse.left, 2)
                    if((leftbound != 0) and (rightbound!= 0)):
                        f20_int += sum(itertools.islice(signalpe, leftbound, min(rightbound, len(signalpe) - 1)))
                    leftbound, rightbound = self.GetIntegrationBounds(int(s1.center_time / 10.0 - len(s1.sum_waveform) / 2.0) + starttime, s1right, pulse.left, 3)
                    if((leftbound != 0) and (rightbound!= 0)):
                        f30_int += sum(itertools.islice(signalpe, leftbound, min(rightbound, len(signalpe) - 1)))
                    leftbound, rightbound = self.GetIntegrationBounds(int(s1.center_time / 10.0 - len(s1.sum_waveform) / 2.0) + starttime, s1right, pulse.left, 4)
                    if((leftbound != 0) and (rightbound!= 0)):
                        f40_int += sum(itertools.islice(signalpe, leftbound, min(rightbound, len(signalpe) - 1)))
                    leftbound, rightbound = self.GetIntegrationBounds(int(s1.center_time / 10.0 - len(s1.sum_waveform) / 2.0) + starttime, s1right, pulse.left, 5)
                    if((leftbound != 0) and (rightbound!= 0)):
                        f50_int += sum(itertools.islice(signalpe, leftbound, min(rightbound, len(signalpe) - 1)))
                    leftbound, rightbound = self.GetIntegrationBounds(int(s1.center_time / 10.0 - len(s1.sum_waveform) / 2.0) + starttime, s1right, pulse.left, 6)
                    if((leftbound != 0) and (rightbound!= 0)):
                        f60_int += sum(itertools.islice(signalpe, leftbound, min(rightbound, len(signalpe) - 1)))
                    leftbound, rightbound = self.GetIntegrationBounds(int(s1.center_time / 10.0 - len(s1.sum_waveform) / 2.0) + starttime, s1right, pulse.left, 7)
                    if((leftbound != 0) and (rightbound!= 0)):
                        f70_int += sum(itertools.islice(signalpe, leftbound, min(rightbound, len(signalpe) - 1)))
                    leftbound, rightbound = self.GetIntegrationBounds(int(s1.center_time / 10.0 - len(s1.sum_waveform) / 2.0) + starttime, s1right, pulse.left, 8)
                    if((leftbound != 0) and (rightbound!= 0)):
                        f80_int += sum(itertools.islice(signalpe, leftbound, min(rightbound, len(signalpe) - 1)))
                    leftbound, rightbound = self.GetIntegrationBounds(int(s1.center_time / 10.0 - len(s1.sum_waveform) / 2.0) + starttime, s1right, pulse.left, 9)
                    if((leftbound != 0) and (rightbound!= 0)):
                        f90_int += sum(itertools.islice(signalpe, leftbound, min(rightbound, len(signalpe) - 1)))
                    leftbound, rightbound = self.GetIntegrationBounds(int(s1.center_time / 10.0 - len(s1.sum_waveform) / 2.0) + starttime, s1right, pulse.left, 10)
                    if((leftbound != 0) and (rightbound!= 0)):
                        f100_int += sum(itertools.islice(signalpe, leftbound, min(rightbound, len(signalpe) - 1)))
                    leftbound, rightbound = self.GetIntegrationBounds(int(s1.center_time / 10.0 - len(s1.sum_waveform) / 2.0) + starttime, s1right, pulse.left, 11)
                    if((leftbound != 0) and (rightbound!= 0)):
                        f110_int += sum(itertools.islice(signalpe, leftbound, min(rightbound, len(signalpe) - 1)))
                    leftbound, rightbound = self.GetIntegrationBounds(int(s1.center_time / 10.0 - len(s1.sum_waveform) / 2.0) + starttime, s1right, pulse.left, 12)
                    if((leftbound != 0) and (rightbound!= 0)):
                        f120_int += sum(itertools.islice(signalpe, leftbound, min(rightbound, len(signalpe) - 1)))
                    leftbound, rightbound = self.GetIntegrationBounds(int(s1.center_time / 10.0 - len(s1.sum_waveform) / 2.0) + starttime, s1right, pulse.left, 13)
                    if((leftbound != 0) and (rightbound!= 0)):
                        f130_int += sum(itertools.islice(signalpe, leftbound, min(rightbound, len(signalpe) - 1)))
                    leftbound, rightbound = self.GetIntegrationBounds(int(s1.center_time / 10.0 - len(s1.sum_waveform) / 2.0) + starttime, s1right, pulse.left, 14)
                    if((leftbound != 0) and (rightbound!= 0)):
                        f140_int += sum(itertools.islice(signalpe, leftbound, min(rightbound, len(signalpe) - 1)))
                    leftbound, rightbound = self.GetIntegrationBounds(int(s1.center_time / 10.0 - len(s1.sum_waveform) / 2.0) + starttime, s1right, pulse.left, 15)
                    if((leftbound != 0) and (rightbound!= 0)):
                        f150_int += sum(itertools.islice(signalpe, leftbound, min(rightbound, len(signalpe) - 1)))
                    leftbound, rightbound = self.GetIntegrationBounds(int(s1.center_time / 10.0 - len(s1.sum_waveform) / 2.0) + starttime, s1right, pulse.left, 16)
                    if((leftbound != 0) and (rightbound!= 0)):
                        f160_int += sum(itertools.islice(signalpe, leftbound, min(rightbound, len(signalpe) - 1)))
                    leftbound, rightbound = self.GetIntegrationBounds(int(s1.center_time / 10.0 - len(s1.sum_waveform) / 2.0) + starttime, s1right, pulse.left, 17)
                    if((leftbound != 0) and (rightbound!= 0)):
                        f170_int += sum(itertools.islice(signalpe, leftbound, min(rightbound, len(signalpe) - 1)))
                    leftbound, rightbound = self.GetIntegrationBounds(int(s1.center_time / 10.0 - len(s1.sum_waveform) / 2.0) + starttime, s1right, pulse.left, 18)
                    if((leftbound != 0) and (rightbound!= 0)):
                        f180_int += sum(itertools.islice(signalpe, leftbound, min(rightbound, len(signalpe) - 1)))
                    leftbound, rightbound = self.GetIntegrationBounds(int(s1.center_time / 10.0 - len(s1.sum_waveform) / 2.0) + starttime, s1right, pulse.left, 19)
                    if((leftbound != 0) and (rightbound!= 0)):
                        f190_int += sum(itertools.islice(signalpe, leftbound, min(rightbound, len(signalpe) - 1)))
                    leftbound, rightbound = self.GetIntegrationBounds(int(s1.center_time / 10.0 - len(s1.sum_waveform) / 2.0) + starttime, s1right, pulse.left, 20)
                    if((leftbound != 0) and (rightbound!= 0)):
                        f200_int += sum(itertools.islice(signalpe, leftbound, min(rightbound, len(signalpe) - 1)))
                    leftbound, rightbound = self.GetIntegrationBounds(int(s1.center_time / 10.0 - len(s1.sum_waveform) / 2.0) + starttime, s1right, pulse.left, 50)
                    if((leftbound != 0) and (rightbound!= 0)):
                        total_int += sum(itertools.islice(signalpe, leftbound, min(rightbound, len(signalpe) - 1)))
            if(total_int == 0):
                continue
            s1.nick = int(s1.center_time / 10.0 - len(s1.sum_waveform) / 2.0) + starttime
            s1.f10 = f10_int/total_int
            s1.f20 = f20_int/total_int
            s1.f30 = f30_int/total_int
            s1.f40 = f40_int/total_int
            s1.f50 = f50_int/total_int
            s1.f60 = f60_int/total_int
            s1.f70 = f70_int/total_int
            s1.f80 = f80_int/total_int
            s1.f90 = f90_int/total_int
            print(s1.f90)
            s1.f100 = f100_int/total_int
            s1.f110 = f110_int/total_int
       	    s1.f120 = f120_int/total_int
       	    s1.f130 = f130_int/total_int
       	    s1.f140 = f140_int/total_int
       	    s1.f150 = f150_int/total_int
       	    s1.f160 = f160_int/total_int
       	    s1.f170 = f170_int/total_int
       	    s1.f180 = f180_int/total_int
       	    s1.f190 = f190_int/total_int
       	    s1.f200 = f200_int/total_int
            found_pulse.clear()
        return event

    def GetIntegrationBounds(self, s1left, s1right, pulseleft, intlength):
        leftbound = 0
        rightbound = 0
        if(s1left >= pulseleft):
            leftbound = s1left - pulseleft
        if(s1left + intlength >= pulseleft):
            rightbound = s1left - pulseleft + intlength
        return (leftbound, rightbound)
