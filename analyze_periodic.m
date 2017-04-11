function [] = analyze_periodic(amplitude, signal_delay, signal_length, period, max_analysis_frequency)
%% Function for compliance with r2015a

    function [deg]  = rad2deg(rad)
        deg = rad / pi * 180;
    end

%% Parameters

sampling_frequency = 1 / period * 2.5 * 1000;
N = 200;

dt = 1/sampling_frequency;
t = 0:dt:period*N-dt;
d = signal_delay+signal_length/2:period:N*period;

NFFT = length(t);

%% Signal

signal = amplitude * pulstran(t,d,'rectpuls', signal_length);
figure;
subplot(2,2,[1,2]);
plot(t,signal,'-','LineWidth',2,'Color','r');
title('Rectangular pulse train');
xlabel('t (s)');
ylabel('U (t)');
axis([0 4*period -max(signal) max(signal)*2]);
grid on;

%% FFT

fft_signal = fft(signal,NFFT);
fft_signal = fftshift(fft_signal);

f = sampling_frequency*(-NFFT/2:NFFT/2-1)/NFFT;

s_f = abs(fft_signal);
s_f = s_f / max(s_f); % Anything less 1% == 0
phi_f = angle(fft_signal);
for ii = 1:length(s_f)
    if(s_f(ii)<0.01)
        s_f(ii) = 0;
        phi_f(ii) = 0;
    end
end
phi_f = rad2deg(phi_f);

subplot(2,2,3);
plot(f,s_f,'-','LineWidth',2,'Color','r');
title('Normalized amplitude spectrum');
xlabel('f (Hz)');
ylabel('|S(f)|');

axis([-max_analysis_frequency max_analysis_frequency 0 1]);
grid on;

subplot(2,2,4);
plot(f,phi_f,'-','LineWidth',2,'Color','b');
title('Phase spectrum');
xlabel('f (Hz)');
ylabel('Phi(f)');

axis([-max_analysis_frequency max_analysis_frequency -180 180]);
grid on;
end