import torch

from .deepindexwindowmemoryentitymodel import DeepIndexWindowMemoryEntityModel
from .indexsmoothedwindowmemoryentitymodel import IndexSmoothedWindowMemoryEntityModel
from .lstmentitymodel import LSTMEntityModel
from .testcharentitymodel import IndexCharsWindowMemoryEntityModel
from .lstmcharentitymodel import LSTMCharsEntityModel

from .indexsplitspansdirectedrelationmodel import IndexSplitSpansDirectedRelationModel
from .indexsplitspansdirectedlabelledrelationmodel import IndexSplitSpansDirectedLabelledRelationModel
from .rulesrelationmodel import RulesRelationModel

from .windowattributemodel import WindowAttributeModel

_classes = [
		DeepIndexWindowMemoryEntityModel,
		IndexSmoothedWindowMemoryEntityModel,
		IndexSplitSpansDirectedRelationModel,
		IndexSplitSpansDirectedLabelledRelationModel,
		LSTMEntityModel,
		LSTMCharsEntityModel,
		RulesRelationModel,
		IndexCharsWindowMemoryEntityModel,
		WindowAttributeModel,
	]

_class_dict = {i.__name__: i for i in _classes}

def load_ann_model(modelfile):
	state = torch.load(modelfile)
	modelclass = _class_dict[state['__model_class']]
	model = modelclass.load_from_state_dict(state)
	return model
