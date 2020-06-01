import os
import pickle
import matplotlib.pyplot as plt
from models import Cache
import numpy as np
import random
from audio_preprocessor import Preprocessor

class DataUtils:

    @staticmethod
    def get_files_list(configuration):
        audio_files_path = configuration.paths().train_data()

        category_size = configuration.data().category_size()
        result = []
        for label in os.listdir(audio_files_path):
            label_catalog = "{}/{}".format(audio_files_path, label)
            count = 0
            for filename in os.listdir(label_catalog):
                # bierzemy tylko pliki wav
                if not filename.endswith("wav"):
                    continue
                # jeżeli w configu zostało ustalone, że bierzemy określoną liczbę plików z każdej kategorii
                if category_size != -1 and count > category_size:
                    continue
                dataEntry = DataEntry("{}/{}".format(label_catalog, filename), label)
                result.append(dataEntry)
                count += 1
        return result

    @staticmethod
    def __get_files_count(path):
        path, dirs, files = next(os.walk(path))
        return len(files)

    @staticmethod
    def __get_train_data_size(path, train_data_proportion):
        path, dirs, files = next(os.walk(path))
        files_count = len(files)
        return int(files_count * train_data_proportion)

    @staticmethod
    def pickle_data(data, path, filename):
        with open(path + "/" + filename, 'wb+') as file:
            pickle.dump(data, file)

    @staticmethod
    def get_object(data):
        return pickle.load(data)

    @staticmethod
    def write_spectogram_to_image(spectrogram, output_filename):
        plt.imsave('%s.png' % output_filename, spectrogram)
        plt.close()

    @staticmethod
    def create_cache_files(configuration, preprocessor):
        train_files, test_files, validation_files = DataUtils.prepare_files_list(configuration)

        DataUtils.create_cache(train_files, preprocessor, configuration.paths().train_cache(), configuration)
        DataUtils.create_cache(test_files, preprocessor, configuration.paths().test_cache(), configuration)
        DataUtils.create_cache(validation_files, preprocessor, configuration.paths().validation_cache(), configuration)

    @staticmethod
    def create_cache(files, preprocessor, path, configuration):
        '''
        Zapisuje pliki na dysku, zawierające obiekt Cache(spektrogramy, etykiety)
        :param files: lista plików ( zawierająca ścieżkę do plików i etykiete)
        :param preprocessor: obiekt zajmujący się tworzeniem spektrogramów
        :param path: ścieżka katalogu wyjściowego
        :param configuration: ustawienia
        :return:
        '''
        batch_size = configuration.data().batch_file_size()
        input_size = configuration.data().input_size()
        num_classes = configuration.settings().num_classes()

        data = []
        targets = []
        batch_number = 0

        for dataEntry in files:
            spectrogram = preprocessor.get_spectrogram(dataEntry.get_path())
            data.append(spectrogram)
            # zamiana nazwy etykiety na tablicę o rozmiarze liczby klas, która zawiera wartość 1 na pozycji odpowiedniej etykiety
            target = DataUtils.__encode(DataUtils.__get_target(dataEntry.get_label()), num_classes)
            targets.append(target)
            # jeżeli lista ma określony rozmiar zapisujemy plik na dysku
            if len(data) == batch_size:
                print("Saving cache " + str(batch_number))
                DataUtils.save_batch(batch_number, path, input_size, data, targets, num_classes)
                batch_number += 1
                data.clear()
                targets.clear()
        # zapisywanie tego co zostało
        DataUtils.save_batch(batch_number, path, input_size, data, targets, num_classes)

    @staticmethod
    def prepare_files_list(configuration):
        # ładowanie listy ścieżek wszystkich plików audio
        files = DataUtils.get_files_list(configuration)
        random.shuffle(files) #mieszanie
        # podział na zbiory
        train_size = configuration.settings().train_size()
        test_size = configuration.settings().test_size()
        train_end_index = int(round(train_size / 100 * len(files)))
        test_end_index = int(round((train_size + test_size) / 100 * len(files)))
        train_list = files[:train_end_index]
        test_list = files[train_end_index:test_end_index]
        validation_list = files[test_end_index:]
        return train_list, test_list, validation_list

    @staticmethod
    def save_batch(batch_number, path, correct_size, data, targets, num_classes):
        # transformacja danych do formy, która może być wykorzystana bezpośrednio na sieci
        data_array = DataUtils.get_array(correct_size, data)
        target_array = DataUtils.get_target_array(num_classes, targets)
        cache = Cache(data_array, target_array)
        # serializacja i zapis na dysku
        DataUtils.pickle_data(cache, path, 'cache' + str(batch_number) + '.pkl')
        return cache

    @staticmethod
    def get_array(input_size, data):
        '''

        :param input_size: rozmiar poprawnego spektrogramu (ustalonego w pliku configuration.cfg)
        :param data: lista spektrogramów
        :return: lista typu ndarray o rozmiarach (liczba plików w partii, x spektrogramu, y spektrogramu, 1).
                Ta tablica nadaje się do podania bezpośrednio na sieć
        '''
        x = input_size[0]
        y = input_size[1]
        length = len(data)
        # rozmiar ndarray jest niemodyfikowalny, więc najpierw trzeba utworzyć odpowiednią tablicę, a później modyfikować
        # jej wartości
        array = np.zeros(shape=[length, x, y, 1])
        correct_size = x * y
        for i in range(len(data)):
            spectrogram = data[i]

            if spectrogram.size != correct_size:
                spectrogram.resize((x, y), refcheck=False)
            # trzeba dodać wymiar do tablicy (1 na końcu)
            reshaped_spectrogram = spectrogram.reshape([x, y, 1])
            array[i] = reshaped_spectrogram
        return array

    @staticmethod
    def get_target_array(num_classes, data):
        array = np.zeros(shape=[len(data), num_classes])
        for i in range(len(data)):
            array[i] = data[i]
        return array

    @staticmethod
    def __encode(target, num_classes):
        targets = np.zeros(shape=[num_classes])
        targets[target] = 1
        return targets



    @staticmethod
    def __get_target(target):
        labels = {
            'yes': 0,
            'no': 1,
            'up': 2,
            'down': 3,
            'left': 4,
            'right': 5,
            'on': 6,
            'off': 7,
            'stop': 8,
            'go': 9
        }
        if target in labels:
            return labels[target]
        return 10  # unknown

    @staticmethod
    def check_and_create_folders(configuration):
        main_folder = configuration.paths().main_folder()
        folders = configuration.paths().get_output_folders()
        if not os.path.isdir(main_folder):
            os.makedirs(main_folder)
        for folder in folders:
            if not os.path.isdir(folder):
                os.makedirs(folder)

    @staticmethod
    def create_cache_if_not_exists(configuration):
        if not DataUtils.__check_cache_exists(configuration):
            print("Create cache")
            preprocessor = Preprocessor()
            DataUtils.create_cache_files(configuration, preprocessor)

    @staticmethod
    def __check_cache_exists(configuration):
        cache_folder = configuration.paths().train_cache()
        return os.listdir(cache_folder) != []


class DataEntry:

    def __init__(self, path, label):
        self.__path = path
        self.__label = label

    def get_path(self):
        return self.__path

    def get_label(self):
        return self.__label
