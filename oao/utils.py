# -*- coding: utf-8 -*-

import json

from ax.modelbridge.generation_strategy import GenerationStrategy
from ax.service.ax_client import ObjectiveProperties
from ax.storage.json_store.decoder import object_from_json
from ax.storage.json_store.encoder import object_to_json


class ConfigEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (ObjectiveProperties)):
            return {"__type": "ObjectiveProperties", "kwargs": o.__dict__}
        elif isinstance(o, (GenerationStrategy)):
            return object_to_json(o)
        else:
            return json.JSONEncoder.encode(self, o)


class ConfigDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        if "__type" not in obj:
            return obj
        obj_type = obj["__type"]
        if obj_type == "ObjectiveProperties":
            return ObjectiveProperties(**obj["kwargs"])
        elif obj_type == "GenerationStrategy":
            return object_from_json(obj)
        else:
            return obj
