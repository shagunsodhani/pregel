from app.utils.constant import GCN, SYMMETRIC


class ModelParams():
    '''
    Class for the params used by the underlying models
    '''

    def __init__(self, flags):
        self.model_name = flags.model_name
        self.norm_mode = None
        self.learning_rate = flags.learning_rate
        self.epochs = flags.epochs
        self.hidden_layer1_size = flags.hidden_layer1_size
        self.dropout = flags.dropout
        self.l2_weight = flags.l2_weight
        self.early_stopping = flags.early_stopping
        self.sparse_features = flags.sparse_features
        self.support_size = flags.support_size
        self.populate_params()

    def populate_params(self):
        '''
        Method to populate all the params for the model
        '''
        if (self.model_name == GCN):
            self.norm_mode = SYMMETRIC


class SparseModelParams():
    '''
    Class for the params that are used when sparse data representation is used.
    '''

    def __init__(self, num_elements, feature_size):
        self.num_elements = num_elements
        self.feature_size = feature_size
