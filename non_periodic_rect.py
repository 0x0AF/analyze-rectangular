import getopt
import numpy as np
import scipy.integrate as integrate
import sys
from matplotlib import pylab, rc

STEPS = 1000


def step(var):
    return (np.sign(var) + 1) * 0.5


def sig(time, amplitude, delay, length):
    return amplitude * (step(time - delay) - step(time - delay - length))


def top_analysis_frequency(length):
    return 2.5 * 1 / length * 2.5


def analyze_non_periodic(amplitude, delay, length):
    s_begin = delay
    s_end = delay + length
    dt = (s_end - s_begin) / STEPS

    strobes = np.linspace(s_begin - dt, s_end + dt, num=STEPS)

    freqs = []
    amps = []
    phs = []
    w = -top_analysis_frequency(length)

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
        phs.append(np.angle(np.array(comp)) if abs(comp) > max(amps) * 0.001 else 0.00)

    _bot_b = amplitude * -0.5
    _top_b = amplitude * 1.5
    _left_b = s_begin - 0.2 * length
    _right_b = s_end + 0.2 * length
    _maj_tick_x = (_right_b - _left_b) / 10.0
    _min_tick_x = (_right_b - _left_b) / 50.0
    _maj_tick_y = (_top_b - _bot_b) / 10.0
    _min_tick_y = (_top_b - _bot_b) / 50.0
    _maj_tick_freq = 1.0 / length
    _min_tick_freq = _maj_tick_freq / 5.0

    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    rc('text', usetex=True)

    f = pylab.figure()

    sbp_1 = f.add_subplot(2, 2, 1)
    sbp_1.set_ylabel('S(t)')
    sbp_1.set_xlabel('t, s')
    sbp_1.set_xlim([_left_b, _right_b])
    sbp_1.set_ylim([_bot_b, _top_b])

    sbp_1.set_xticks(np.arange(_left_b, _right_b + _maj_tick_x, _maj_tick_x))
    sbp_1.set_xticks(np.arange(_left_b, _right_b + _min_tick_x, _min_tick_x), minor=True)
    sbp_1.set_yticks(np.arange(_bot_b, _top_b + _maj_tick_y, _maj_tick_y))
    sbp_1.set_yticks(np.arange(_bot_b, _top_b + _min_tick_y, _min_tick_y), minor=True)

    sbp_1.grid(which='minor', alpha=0.2)
    sbp_1.grid(which='major', alpha=0.75)

    pylab.title(r'Signal')
    pylab.plot(strobes, sig(strobes, amplitude, delay, length))

    sbp_2 = f.add_subplot(2, 2, 3)
    sbp_2.set_ylabel(r'S(f)')
    sbp_2.set_xlabel(r'f, Hz')
    sbp_2.set_xlim([min(freqs), max(freqs)])
    sbp_2.set_ylim([0, max(amps)])

    sbp_2.set_xticks(np.arange(np.ceil(min(freqs)), np.floor(max(freqs)) + _maj_tick_freq, _maj_tick_freq))
    sbp_2.set_xticks(np.arange(np.ceil(min(freqs)), np.floor(max(freqs)) + _min_tick_freq, _min_tick_freq), minor=True)
    sbp_2.set_yticks(np.arange(0, max(amps) + _maj_tick_y, _maj_tick_y))
    sbp_2.set_yticks(np.arange(0, max(amps) + _min_tick_y, _min_tick_y), minor=True)

    sbp_2.grid(which='minor', alpha=0.2)
    sbp_2.grid(which='major', alpha=0.75)

    pylab.title(r'Amplitude spectrum')
    pylab.plot(freqs, amps)

    sbp_3 = f.add_subplot(2, 2, 4)
    degs = sbp_3.twinx()

    sbp_3.set_ylabel(r'$\Phi$, rad')
    sbp_3.set_xlabel(r'f, Hz')
    sbp_3.set_xlim([min(freqs), max(freqs)])
    sbp_3.set_ylim([-np.pi, np.pi])

    sbp_3.set_xticks(np.arange(np.ceil(min(freqs)), np.floor(max(freqs)) + _maj_tick_freq, _maj_tick_freq))
    sbp_3.set_xticks(np.arange(np.ceil(min(freqs)), np.floor(max(freqs)) + _min_tick_freq, _min_tick_freq), minor=True)
    sbp_3.set_yticks(np.arange(-np.pi, np.pi + np.pi / 4.0, np.pi / 4.0))
    sbp_3.set_yticks(np.arange(-np.pi, np.pi + np.pi / 20.0, np.pi / 20.0), minor=True)
    sbp_3.set_yticklabels([r'$-\Pi$', r'$-3\Pi/4$', r'$-\Pi/2$',
                           r'$-\Pi/4$', r'$0$', r'$\Pi/4$',
                           r'$\Pi/2$', r'$3\Pi/4$', r'$\Pi$'])

    degs.set_ylabel(r'$\Phi$, deg')
    degs.set_ylim([-180, 180])
    degs.set_yticks(np.arange(-180.0, 180.0 + 45.0, 45.0))
    degs.set_yticks(np.arange(-180.0, 180.0 + 15.0, 15.0), minor=True)

    sbp_3.grid(which='minor', alpha=0.2)
    sbp_3.grid(which='major', alpha=0.75)

    pylab.title(r'Phase spectrum')
    sbp_3.plot(freqs, phs)
    # degs.plot(freqs, np.degrees(phs))

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
