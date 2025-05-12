from .GA import GA, GAR, GRDA
from .FT import FT
from .retrain import retrain
from .impl import load_unlearn_checkpoint, save_unlearn_checkpoint
from .RELOAD import RELOAD
from .SSD import SSD
from .SCRUB import SCRUB
from .euk import EU_k
from .cuk import CU_k


def raw(data_loaders, model, criterion, args, mask=None):
    pass


def get_unlearn_method(name):
    """method usage:

    function(data_loaders, model, criterion, args)"""
    if name == "raw":
        return raw
    elif name == "GA":
        return GA
    elif name == "GAR":
        return GAR
    elif name == "GRDA":
        return GRDA
    elif name == "FT":
        return FT
    elif name == "retrain":
        return retrain
    elif name == "RELOAD":
        return RELOAD
    elif name == "SSD":
        return SSD
    elif name == "SCRUB":
        return SCRUB
    elif name == "EUk":
        return EU_k
    elif name == "CUk":
        return CU_k
    else:
        raise NotImplementedError(f"Unlearn method {name} not implemented!")
