import os
import random

import tensorflow as tf
from tensorflow.train import Saver

from data_utils import DataUtils
from tf_model import Model
import datetime

from sklearn.utils import shuffle
from model_config import ModelConfig
import matplotlib
import numpy as np
import itertools
import matplotlib.pyplot as plt
import io


# TODO: Refactor!!!
class Recognition:

    def __init__(self, configuration):
        self.__configuration = configuration

    def train(self):
        # zapobieganie wyświetlanie notacji naukowej (to z e) przy print
        np.set_printoptions(suppress=True)
        # pobranie ustawień
        train_folder = self.__configuration.paths().train_cache()
        validation_folder = self.__configuration.paths().validation_cache()
        train_files = os.listdir(train_folder)
        validation_files = os.listdir(validation_folder)
        batch_size = self.__configuration.data().batch_size()
        input_size = self.__configuration.data().input_size()
        device = self.__configuration.settings().device()
        iterations = self.__configuration.settings().iterations()
        # do folderu z modelem dodajemy datę rozpoczęcia uczenia
        start_date = "model_1"
        output_model_folder = self.__configuration.paths().model() + "_" + start_date
        learning_rate = self.__configuration.settings().learning_rate()
        num_classes = self.__configuration.settings().num_classes()

        # utworzenie elementów tensorflow, do których wstawiane będa dane treningowe
        train_x_input, train_y_input = self.__create_placeholders(batch_size, input_size, num_classes)
        learning_rate_placeholder = tf.placeholder(tf.float32, None)
        tf_session = self.__create_tf_session()

        with tf.variable_scope("scope", reuse=tf.AUTO_REUSE):
            model = Model.create(self.__configuration, train_x_input)
            # utworzenie funkcji potrzebnych do oceny modelu
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=train_y_input))
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_placeholder).minimize(cost)
            correct_predictions = tf.equal(tf.argmax(model, 1), tf.argmax(train_y_input, 1))
            accuracy_op = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
            tf.summary.scalar('accuracy', accuracy_op)
            merged = tf.summary.merge_all()
            test_writer = tf.summary.FileWriter(self.__configuration.paths().output(), tf_session.graph)
            # obiekt zapisujący model
            saver = Saver()
            # inicjalizacja sesji tensorflow
            tf_session.run(tf.global_variables_initializer())
            # tablica przechowująca wyniki poszczególnych iteracji
            accuracies = []
            with tf.device('/device:' + device):
                counter = 0
                for iteration in range(iterations):
                    train_accuracy = 0.0
                    batch_counter = 0
                    print("iteration " + str(iteration))
                    # mieszanie kolejności odczytu plików z dysku
                    random.shuffle(train_files)
                    for batch_file in train_files:
                        # odczyt partii z dysku
                        data = self.__get_data_from_batch_file(batch_file, train_folder)
                        # mieszanie tablic z danymi i etykietami
                        shuffled_X, shuffled_y = shuffle(data.get_X(), data.get_y())
                        # tworzenie i iteracja po partiach danych
                        for batch_X, batch_y in self.__get_batchs_list(shuffled_X, shuffled_y, batch_size):
                            # karmienie sieci
                            _, acc = tf_session.run([optimizer, accuracy_op], feed_dict={train_x_input: batch_X, train_y_input: batch_y,
                                                                 learning_rate_placeholder: learning_rate})
                            train_accuracy += acc
                            batch_counter += 1
                    # walidacja
                    accuracy, confusion_matrix = self.__validate(accuracy_op, tf_session, train_x_input, train_y_input, validation_files,
                                               validation_folder, batch_size, model, merged, test_writer, counter)
                    # save train accuracy
                    train_accuracy = train_accuracy/batch_counter
                    train_accuracy_summary = tf.Summary(value=[
                        tf.Summary.Value(tag="train_accuracy", simple_value=train_accuracy),
                    ])
                    test_writer.add_summary(train_accuracy_summary, iteration)
                    if accuracy > 0.8: # TODO można ustawić inny warunek
                        print("Zmiana współczuynnika uczenia")
                        learning_rate = self.__configuration.settings().second_learning_rate()
                    accuracies.append(accuracy)
                    counter = counter + 1
                # self.test_confustion_matrix(tf_session, validation_files, validation_folder, correct_predictions, model) # TODO zastanowić się, czy to correct_predictions tutaj powinno zostac przekazane
                self.__save_results(accuracies, output_model_folder, saver, tf_session)
                test_writer.close()

    def __save_results(self, accuracies, output_model_folder, saver, tf_session):
        # utworzenie modelu z datą w nazwie ( data wstawiana jest poza tą metodą)
        os.makedirs(output_model_folder)
        # zapisanie modelu
        saver.save(tf_session, output_model_folder + '/model.ckpt')
        # zapisanie listy wyników w poszczególnych iteracji
        with open(output_model_folder + '/accuracy.txt', 'w+') as file:
            for acc in accuracies:
                file.write(str(acc) + '\n')
        # zapisanie parametrów wykorzystanych do uczenia modelu
        with open(output_model_folder + "/settings.txt", 'w+') as file:
            for field, value in ModelConfig().get_config().items():
                file.write(field + ": " + str(value) + '\n')

    def __validate(self, accuracy_op, tf_session, train_x_input, train_y_input, validation_files, validation_folder,
                   batch_size, model, merged, test_writer, counter):
        result = 0.0
        matrix_result = np.zeros(shape=(11,11)) # TODO
        batch_count = 0

        for batch_file in validation_files:
            data = self.__get_data_from_batch_file(batch_file, validation_folder)
            shuffled_X, shuffled_y = shuffle(data.get_X(), data.get_y())
            for batch_X, batch_y in self.__get_batchs_list(shuffled_X, shuffled_y, batch_size):
                # liczenie poprawności na pojedyńczej partii
                acc, pred =  tf_session.run([accuracy_op, model], feed_dict={train_x_input: batch_X,
                                                                         train_y_input: batch_y})
                result += acc

                # liczenie macierzy pomyłek
                predicted_batch = tf.argmax(pred, axis=1)
                real_label_batch = tf.argmax(batch_y, axis=1)
                confusion_matrix = tf.confusion_matrix(labels=real_label_batch, predictions=predicted_batch, num_classes=self.__configuration.settings().num_classes())
                matrix = tf_session.run(confusion_matrix)
                matrix_result += matrix

                batch_count += 1
        # średnia
        confusion_matrix_summary = self.plot_confusion_matrix(matrix_result)
        test_writer.add_summary(confusion_matrix_summary.eval(session=tf_session), counter)
        accuracy = result / batch_count
        accuracy_summary = tf.Summary(value=[
                        tf.Summary.Value(tag="accuracy", simple_value=accuracy),
                    ])
        test_writer.add_summary(accuracy_summary, counter)
        print(accuracy)
        print(matrix_result)
        return accuracy, matrix_result

    def plot_confusion_matrix(self, confusion_matrix):
        plt.imshow(confusion_matrix, interpolation='nearest', cmap='Oranges')
        plt.title("Macierz błędów")
        plt.colorbar()
        # TODO wstawić etykiety
        # tick_marks = np.arange(len(classes))
        # plt.xticks(tick_marks, classes, rotation=45)
        # plt.yticks(tick_marks, classes)

        # fmt = '.2f' if normalize else 'd'
        fmt = '.2f'
        thresh = confusion_matrix.max() / 2.
        for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
            plt.text(j, i, format(confusion_matrix[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if confusion_matrix[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)
        plt.figure()
        confusion_matrix_summary = tf.summary.image('confusion_matrix', image, max_outputs=1)
        return confusion_matrix_summary


    def test_confustion_matrix(self,session, files, folder, predictions, model):
        placeholder = tf.placeholder(tf.float32, shape=(self.__configuration.settings().num_classes(),1))
        predicted_batch = tf.argmax(predictions, axis=1)
        real_label_batch = tf.argmax(placeholder, axis=1)
        num_classes = self.__configuration.settings().num_classes()
        matrix = None
        confusion_batch = tf.confusion_matrix(labels=real_label_batch, predictions= predicted_batch, num_classes=num_classes)
        for file in files:
            data = self.__get_data_from_batch_file(file, folder)
            for record in data.get_targets():
                current_matrix =  session.run(confusion_batch, feed_dict={placeholder: record})
                if not matrix:
                    matrix = current_matrix
                else:
                    matrix += current_matrix
        print(matrix)
        return matrix


    def __get_batchs_list(self, data, targets, batch_size):
        '''
        Zwraca listę danych podzielonych w partie o podanym rozmiarze. Ostatnie elementy, które nie zapełniły partii nie są
        zwracane.
        :param data: duża tablica danych odczytana z dysku
        :param targets: duża tablica etykiet odczytana z dysku
        :param batch_size: rozmiar partii
        :return: lista tablic o rozmiarze batch_size
        '''
        number_of_batch = int(len(data) / batch_size)
        batches = []
        for number in range(number_of_batch):
            start_index = number * batch_size
            end_index = number * batch_size + batch_size
            X = data[start_index: end_index, :]
            y = targets[start_index: end_index, :]
            batch_data = (X, y)
            batches.append(batch_data)
        return batches

    @staticmethod
    def __create_tf_session():
        config = tf.ConfigProto(allow_soft_placement=True)
        session = tf.Session(config=config)
        return session

    @staticmethod
    def __get_accuracy_op(model, y):
        correct_predictions = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
        accuracy_op = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        return accuracy_op

    @staticmethod
    def __get_accuracy(session, accuracy_op, test_X, test_y, x_test_placeholder, y_test_placeholder):
        return session.run([accuracy_op], feed_dict={
            x_test_placeholder: test_X, y_test_placeholder: test_y
        })

    @staticmethod
    def __create_placeholders(batch_size, input_size, num_classes):
        x_input_shape = (batch_size, input_size[0], input_size[1], 1)
        y_input_shape = (batch_size, num_classes)
        x_input = tf.placeholder(tf.float32, shape=x_input_shape)
        y_input = tf.placeholder(tf.float32, shape=y_input_shape)
        return x_input, y_input

    @staticmethod
    def __get_data_from_batch_file(batch_filename, cache_folder):
        file = open(cache_folder + "/" + batch_filename, 'rb')
        data = DataUtils.get_object(file)
        return data
