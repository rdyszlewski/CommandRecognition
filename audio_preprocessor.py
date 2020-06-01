# math
import numpy as np
from scipy import signal
from scipy.io import wavfile

# visualisation
import matplotlib.pyplot as plt
from models import CommandData


class Preprocessor:

    def get_audio(self, filename):
        '''
        Convert wav file array of samples
        :param filename: path to wav file
        :return: tuple(sample_rate, samples)
        '''
        # sample_rate, samples
        return wavfile.read(filename)

    def get_spectogram(self, filename):
        sample_rate, samples = self.get_audio(filename)

    def get_spectrogram(self, filename, window_size=20, step_size=10, eps=1e-10):
        sample_rate, samples = self.get_audio(filename)
        return self.get_spectrogram_data2(samples, sample_rate, window_size, step_size, eps)[2]

    def get_spectrogram_data(self, audio, sample_rate, window_size=20, step_size=10):
        nperseg = int(round(window_size * sample_rate / 1e3))
        noverlap = int(round(step_size * sample_rate / 1e3))
        freqs, times, spectrogram_data = signal.spectrogram(audio, fs=sample_rate, window="hann", nperseg=nperseg,
                                                            noverlap=noverlap, detrend=False)
        return freqs, times, spectrogram_data

    def get_spectrogram_data2(self, audio, sample_rate, window_size=20, step_size=10, eps=1e-10):
        nperseg = int(round(window_size * sample_rate / 1e3))
        noverlap = int(round(step_size * sample_rate / 1e3))
        freqs, times, spectrogram_data = signal.spectrogram(audio, fs=sample_rate, window="hann", nperseg=nperseg,
                                                            noverlap=noverlap, detrend=False)
        return freqs, times, np.log(spectrogram_data.T.astype(np.float32) + eps)

    def show_spectogram(self, filename):
        sample_rate, samples = self.get_audio(filename)
        freqs, times, spectrogram = self.get_spectrogram_data2(samples, sample_rate)

        figure = plt.figure(figsize=(14, 8))
        raw_plot = figure.add_subplot(211)
        raw_plot.set_title('Raw wave of ' + filename)
        raw_plot.set_ylabel('Amplitude')

        raw_plot.plot(samples)

        spectrogram_plot = figure.add_subplot(212)
        # spectrogram_plot.imshow(spectrogram.T, aspect='auto', origin='lower',
        #            extent=[times.min(), times.max(), freqs.min(), freqs.max()])

        # spectrogram_plot.set_yticks(freqs[::16])
        # spectrogram_plot.set_xticks(times[::16])
        spectrogram_plot.set_title('Spectrogram of ' + filename)
        spectrogram_plot.set_ylabel('Freqs in Hz')
        spectrogram_plot.set_xlabel('Seconds')
        # spectrogram_plot.pcolormesh(times, freqs, spectrogram)
        spectrogram_plot.imshow(spectrogram.T, aspect='auto', origin='lower')
        plt.show()

    def get_audio_objects(self, paths_filename):
        result = []
        file = open(paths_filename, 'r')
        for line in file:
            path, label = line.strip().split('\t')
            print(path)
            sample_rate, audio = self.get_audio(path)
            freqs, times, spectrogram = self.get_spectrogram_data(audio, sample_rate)
            model = CommandData(label, spectrogram)
            result.append(model)
        return result
