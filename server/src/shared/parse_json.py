from bson import ObjectId
import json


def parse_json(data):
    def custom_json_encoder(obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    # Use custom encoder to handle ObjectId    # Use custom encoder to handle ObjectId
    res = json.loads(json.dumps(data, default=custom_json_encoder))
    res['id'] = res.pop('_id')

    return res
