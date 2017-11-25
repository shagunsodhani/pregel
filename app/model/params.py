from app.utils.constant import GCN, SYMMETRIC, GCN_POLY


class ModelParams():
    '''
    Class for the params used by the underlying models
    '''

    def __init__(self, flags):
        self.model_name = flags.model_name
        self.learning_rate = flags.learning_rate
        self.epochs = flags.epochs
        self.hidden_layer1_size = flags.hidden_layer1_size
        try:
            self.hidden_layer2_size = flags.hidden_layer2_size
        except AttributeError:
            self.hidden_layer2_size = None
        self.dropout = flags.dropout
        self.l2_weight = flags.l2_weight
        self.early_stopping = flags.early_stopping
        self.sparse_features = flags.sparse_features
        try:
            self.support_size = flags.poly_degree + 1
        except AttributeError:
            self.support_size = 1
        self.norm_mode = SYMMETRIC
        self.populate_params()

    def populate_params(self):
        '''
        Method to populate all the params for the model
        '''

        if (self.model_name != GCN_POLY):
            self.support_size = 1


class SparseModelParams():
    '''
    Class for the params that are used when sparse data representation is used.
    '''

    def __init__(self, num_elements, feature_size):
        self.num_elements = num_elements
        self.feature_size = feature_size

class AutoEncoderModelParams():
    '''
    Class for the params that are used by the AutoEncoder models - gcn_ae and gcn_vae.
    '''

    def __init__(self, positive_sample_weight):
        self.positive_sample_weight = positive_sample_weight