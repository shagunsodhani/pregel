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
        self.hided_layer1_size = flags.hidden_layer1_size
        self.dropout = flags.dropout
        self.l2_weight = flags.l2_weight
        self.early_stopping = flags.early_stopping
        self.populate_params()

    def populate_params(self):
        '''
        Method to populate all the params for the model
        '''
        if(self.model_name == GCN):
            self.norm_mode = SYMMETRIC

