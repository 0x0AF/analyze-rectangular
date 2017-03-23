import getopt
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
import sys

STEPS = 1000


def step(var):
    return 0.5 * (np.sign(var) + 1)


def signal(time, amplitude, delay, length):
    return amplitude * (step(time - delay) - step(time - delay - length))


def top_analysis_frequency(length):
    return 2.5 * 1 / length * 5


def analyze_non_periodic(amplitude, delay, length):
    sig_time_begin = delay
    sig_time_end = delay + length

    freqs = []
    amps = []
    phs = []
    w = 0.0

    while w < top_analysis_frequency(length):
        w += top_analysis_frequency(length) / STEPS
        comp = integrate.quad(lambda x: signal(x, amplitude, delay, length) * np.exp(-2 * np.pi * 1j * x * w),
                              sig_time_begin,
                              sig_time_end)[0]
        freqs.append(w)
        amps.append(abs(comp))
        phs.append(np.angle(comp))

    # TODO: subplot, add axis labels

    plt.plot(freqs, amps)
    plt.show()

    plt.plot(freqs, phs)
    plt.show()


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
