import json

from collections import OrderedDict


def read_json_file(json_file):
    """
    Read a given JSON file.
    
    Args:
        json_file: The JSON file to read to.

    Returns:
        A JSON file.
        
    """
    
    with open(json_file, "r") as file:
        entity = json.load(file)
    for gt_category in entity.keys():
        gt_text_category = entity[gt_category]
        if len(gt_text_category) == 0:
            return None
    return entity


def reorder_json_file(old_entity, old_ordered_keys: list):
    """
    Re-order the keys in the JSON file.
    
    Args:
        old_entity: The old JSON file.
        old_ordered_keys: The list of keys in the old JSON file.

    Returns:
        A new JSON file with keys re-ordered.
        
    """
    
    new_json_file = OrderedDict()
    for old_key in old_ordered_keys:
        new_json_file[old_key] = old_entity.get(old_key, "none")
    return new_json_file
