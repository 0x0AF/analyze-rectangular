import getopt
import numpy as np
import scipy.integrate as integrate
import sys
from matplotlib import pylab

from enum import Enum


class PMeasures(Enum):
    RADIANS = 0
    DEGREES = 1


STEPS = 1000
P_MEASURE = PMeasures.RADIANS


def step(var):
    return (np.sign(var) + 1) * 0.5


def sig(time, amplitude, delay, length):
    return amplitude * (step(time - delay) - step(time - delay - length))


def top_analysis_frequency(length):
    return 2.5 * 1 / length * 5


def analyze_non_periodic(amplitude, delay, length):
    s_begin = delay
    s_end = delay + length
    dt = (s_end - s_begin) / STEPS

    strobes = np.linspace(s_begin - dt, s_end + dt, num=STEPS)

    freqs = []
    amps = []
    phs = []
    w = 0.0

    while w < top_analysis_frequency(length):
        w += top_analysis_frequency(length) / STEPS
        comp_re = integrate.quad(lambda x: sig(x, amplitude, delay, length) * np.real(np.exp(-2 * np.pi * 1j * x * w)),
                                 s_begin,
                                 s_end)[0]
        comp_im = integrate.quad(lambda x: sig(x, amplitude, delay, length) * np.imag(np.exp(-2 * np.pi * 1j * x * w)),
                                 s_begin,
                                 s_end)[0]
        comp = comp_re + comp_im * 1j

        freqs.append(w)
        amps.append(abs(comp))
        phs.append(np.angle(comp))

    f = pylab.figure()

    sbp_1 = f.add_subplot(2, 2, 1)
    sbp_1.set_ylabel('S(t)')
    sbp_1.set_xlabel('Time, s')

    pylab.title('Signal')
    pylab.plot(strobes, sig(strobes, amplitude, delay, length))

    sbp_2 = f.add_subplot(2, 2, 3)
    sbp_2.set_ylabel('Amplitude')
    sbp_2.set_xlabel('Frequency, Hz')

    pylab.title('Amplitude spectrum')
    pylab.plot(freqs, amps)

    sbp_3 = f.add_subplot(2, 2, 4)
    sbp_3.set_ylabel('Phase, radians' if P_MEASURE == PMeasures.RADIANS else 'Phase, degrees')
    sbp_3.set_xlabel('Frequency, Hz')

    pylab.title('Phase spectrum')
    pylab.plot(freqs, phs if P_MEASURE == PMeasures.RADIANS else np.degrees(phs))

    pylab.show()


def main(argv):
    amplitude = 0.0
    delay = 0.0
    length = 0.0

    try:
        opts, args = getopt.getopt(argv, "",
                                   ["amplitude=", "delay=", "length="])
    except getopt.GetoptError:
        print 'non_periodic_rect.py --amplitude %FLOAT% --delay %FLOAT% --length %FLOAT%'
        sys.exit(2)

    for opt, arg in opts:
        if opt == '--amplitude':
            amplitude = float(arg)
        if opt == '--delay':
            delay = float(arg)
        if opt == '--length':
            length = float(arg)

    analyze_non_periodic(amplitude, delay, length)


if __name__ == "__main__":
    main(sys.argv[1:])
