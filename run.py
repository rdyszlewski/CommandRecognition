from configuration import Configuration
from recognition import Recognition
from tensorflow.python.client import device_lib
from data_utils import DataUtils

def run():
    print(device_lib.list_local_devices())
    configuration = Configuration('configuration/configuration.cfg')

    DataUtils.check_and_create_folders(configuration)
    DataUtils.create_cache_if_not_exists(configuration)

    recognition = Recognition(configuration)
    recognition.train()


run()
