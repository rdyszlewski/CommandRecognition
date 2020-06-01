from configparser import ConfigParser
import os

class Configuration:

    def __init__(self, configuration_filename):
        PATHS = "paths"
        DATA = 'data'
        SETTINGS = "settings"

        parser = ConfigParser()
        dirname =  os.path.dirname(__file__)
        print(dirname)
        parser.read(dirname + '/' + configuration_filename)
        self.__paths = PathConfiguration(parser, PATHS)
        self.__data = DataConfiguration(parser, DATA)
        self.__settings = SettingsConfiguration(parser, SETTINGS)

    def paths(self):
        return self.__paths

    def data(self):
        return self.__data

    def settings(self):
        return self.__settings

    def model(self):
        return self.__model


class PathConfiguration:

    def __init__(self, parser, header):
        self.__output_folder = parser.get(header, "output_folder")
        self.__data_folder = parser.get(header, "data_folder")
        self.__train_data = parser.get(header, "train_data")
        self.__test_data = parser.get(header, "test_data")
        self.__train_cache = parser.get(header, 'train_cache')
        self.__test_cache = parser.get(header, 'test_cache')
        self.__validation_cache = parser.get(header, 'validation_cache')
        self.__output = parser.get(header, 'output')
        self.__model = parser.get(header, 'model')

    def main_folder(self):
        return self.__output_folder

    def train_data(self):
        return self.__data_folder + self.__train_data

    def test_data(self):
        return self.__data_folder + self.__test_data

    def train_cache(self):
        return self.__output_folder + self.__train_cache

    def test_cache(self):
        return self.__output_folder + self.__test_cache

    def validation_cache(self):
        return self.__output_folder + self.__validation_cache

    def output(self):
        return self.__output_folder + self.__output

    def model(self):
        return self.__output_folder + self.__model

    def get_output_folders(self):
        return [self.train_cache(), self.test_cache(), self.validation_cache(), self.output(), self.model()]


class DataConfiguration:

    def __init__(self, parser, header):
        self.__category_size = parser.get(header, "category_size")
        self.__batch_size = parser.get(header, "batch_size")
        self.__batch_file_size = parser.get(header, "batch_file_size")
        input_size = parser.get(header, 'input_size')
        x, y = input_size.split(',')
        self.__input_size = [int(x), int(y)]

    def category_size(self):
        return int(self.__category_size)

    def batch_size(self):
        return int(self.__batch_size)

    def batch_file_size(self):
        return int(self.__batch_file_size)

    def input_size(self):
        return self.__input_size


class SettingsConfiguration:

    def __init__(self, parser, header):
        data_proportion = parser.get(header, 'data_proportion')
        train, test, validation = self.__parse_data_proportion(data_proportion)
        self.__train_size = train
        self.__validation_size = validation
        self.__test_size = test
        self.__iterations = parser.get(header, 'iterations')
        self.__learning_rate = parser.get(header, 'learning_rate')
        self.__second_learning_rate = parser.get(header, 'second_learning_rate')
        self.__dropout = parser.get(header, 'dropout')
        self.__num_classes = parser.get(header, 'num_classes')
        self.__device = parser.get(header, 'device')

    def __parse_data_proportion(self, data_proportion):
        proportion = data_proportion.split(":")
        return int(proportion[0]), int(proportion[1]), int(proportion[2])

    def train_size(self):
        return self.__train_size

    def validation_size(self):
        return self.__validation_size

    def test_size(self):
        return self.__test_size

    def iterations(self):
        return int(self.__iterations)

    def learning_rate(self):
        return float(self.__learning_rate)

    def second_learning_rate(self):
        return float(self.__second_learning_rate)

    def dropout(self):
        return float(self.__dropout)

    def num_classes(self):
        return int(self.__num_classes)

    def device(self):
        return self.__device
