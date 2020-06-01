
class CommandData:

    def __init__(self,label, spectrogram, frequencies = None, times = None):
        self.__label = label
        self.__sample_rate = 0
        self.__frequencies = frequencies
        self.__times = times
        self.__spectrogram = spectrogram

    def get_label(self):
        return [1]

    def get_sample_rate(self):
        return self.__sample_rate

    def get_frequencies(self):
        return self.__frequencies

    def get_times(self):
        return self.__times

    def get_spectrogram(self):
        return self.__spectrogram


class Cache:

    def __init__(self, X, y):
        self.__X = X
        self.__y = y

    def get_X(self):
        return self.__X

    def get_y(self):
        return self.__y

