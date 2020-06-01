

class ModelConfig:

    conv_1_filters = 16
    conv_1_size = 10
    conv_1_padding = 'valid'
    conv_1_activation = "relu"
    pool_1_size = 3
    conv_2_filters = 32
    conv_2_size = 7
    conv_2_padding = 'valid'
    conv_2_activation = "relu"
    pool_2_size = 2
    full_size = 512

    def get_config(self):
        '''
        Metoda zwracająca listę wszystkich elementów klasy, oprócz tej metody.
        :return: słownik {pole: wartość}
        '''
        return {k:v for k,v in ModelConfig.__dict__.items() if not k.startswith("get")}