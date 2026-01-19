import math

import ujson
from boltons.iterutils import remap


class CaptionProcessor:
    
    @classmethod
    def filter_empty_fields(cls, p, k, v):
        is_none = v is None
        is_empty_string = isinstance(v, str) and v == ""
        is_empty_dict = isinstance(v, dict) and not v
        is_empty_list = isinstance(v, list) and not v
        is_nan = isinstance(v, float) and math.isnan(v)   
        if is_none or is_empty_string or is_empty_list or is_empty_dict:
            return False
        if is_nan:
            return False
        return True

    @classmethod
    def prepare_clean_caption(cls, json_dump: dict) -> str:
        clean_caption_dict = remap(json_dump, visit=CaptionProcessor.filter_empty_fields)
        enriched_dict = cls.add_aesthetics_scores(clean_caption_dict)

        return ujson.dumps(enriched_dict, escape_forward_slashes=False)
    
    @classmethod
    def add_aesthetics_scores(cls, data_dict: dict) -> dict:
        scores = {
            "preference_score": "very high",
            "aesthetic_score": "very high"
        }
        
        if "aesthetics" not in data_dict:
            data_dict["aesthetics"] = scores
        else:
            if not isinstance(data_dict["aesthetics"], dict):
                data_dict["aesthetics"] = {}
            data_dict["aesthetics"].update(scores)
            
        return data_dict