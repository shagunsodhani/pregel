from app.utils.constant import FF, GCN, GCN_POLY

from app.model.ff_model import Model as ff_model
from app.model.gcn_model import Model as gcn_model

def select_model(model_name):
    if(model_name == FF):
        return ff_model
    elif(model_name == GCN or model_name == GCN_POLY):
        return gcn_model
    else:
        return None