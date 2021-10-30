import regex
import numpy as np

from collections import OrderedDict
from keyword_information_extraction.data.datasets.sroie2019.variables import SROIE_DATE_PATTERN


def clean_company(textline: str):
    """
    Given a string line, this function will attempt to remove unwanted string that lowers the F1-score.

    Args:
        textline: The string line.

    Returns:
        A cleaned company.
        
    """
    
    # Take only unique words in a given text line.
    # If this text line contains the character ', it is replaced by an empty space.
    textline = " ".join(OrderedDict.fromkeys(textline.replace("'", "").strip().split()))
    
    # A case where the regex below will fail.
    cond = bool(regex.search(r"[A-Z]+[0-9]+", textline.strip()))
    if cond:
        return textline
    
    matched = regex.search(r"(\d+[^0-9]*[A-Z]+)$", textline.strip())
    if matched is None:
        matched = regex.search(r"\(\d+[^0-9]*[A-Z]+\)$", textline.strip())
        if matched is None:
            matched = regex.search(r"\([A-Z]{1,}$", textline.strip())
            if matched is None:
                return textline.strip()
    idx = textline.find(matched.group())
    textline = textline[:idx]
    return textline.strip()


def clean_address(textline: str):
    """
    As some receipts have the phone number on the same line as the address,
    this function will attempt to remove unwanted phone number that lowers the F1-score.

    Args:
        textline: The string line to clean to.

    Returns:
        A cleaned address.
        
    """
    sub_str = "TEL"
    idx = textline.find(sub_str)
    if idx != -1:
        textline = textline[:idx]
    textline = regex.sub(r"(\d+[\-][^a-zA-Z]*)$", "", textline.strip())
    return textline.strip()


def extract_date(textline: str):
    """
    Given a string line, this function will extract the date based on a regex.

    Args:
        textline: The string line for which an attempt to extract a date will be performed.

    Returns:
        A string date based on a regex. Otherwise empty string.
    """
    # First we remove the datetime.
    textline = regex.sub(r"\d+[\:]\d+[\:]*[0-9]*[a-zA-Z\s]*", "", textline.strip())
    
    # Then, we try to match a date.
    matched = regex.search(SROIE_DATE_PATTERN, textline.strip())
    return matched.group().strip() if matched is not None else ""


def extract_total(textline: str):
    """
    Given a string line, this function will extract a float number based on a regex.
    
    Args:
        textline: The string line for which an attempt to extract a float number will be performed.

    Returns:
        A string float number based on a regex. Otherwise empty string.
    """
    matched = regex.search(r"([$\+\-]|[RM\s])*(\d+\.(\d{2}|\d))", textline.strip())
    return matched.group().strip() if matched is not None else ""


def convert_predictions_to_dict(class_dict: dict, raw_texts: str, predicted_classes, probabilities):
    """
    Convert the model's predictions into a dictionary.

    Args:
        class_dict: The dictionary of classes.
        raw_texts: The raw texts, i.e., texts that are not one-hot encoded.
        predicted_classes: The model's predicted classes.
        probabilities:  The model's probabilities.

    Returns:
        A dictionary where keys are the category (company, address, ...) and
        the values are the corresponding raw text associated to a given category.
    """
    
    # We do not need the "none" class.
    results = {class_name: ("", 0.0) for class_name in class_dict.keys() if class_name != "none"}
    categories = list(results.keys())
    
    # The idea of the list below is to get the span of indices of each category
    # For instance, the address category lies within multiple indices (between 2 and 6 indices)
    seps = [0] + (np.nonzero(np.diff(predicted_classes))[0] + 1).tolist() + [len(predicted_classes)]
    
    # First, we identify the category predicted by the model.
    for i in range(len(seps) - 1):
        predicted_class = predicted_classes[seps[i]] - 1
        if predicted_class == -1:  # Skip the non-class label
            continue
        new_category = categories[predicted_class]
        new_probability = max(probabilities[seps[i]: seps[i + 1]])
        _, best_probability_so_far = results[new_category]
        if new_probability > float(best_probability_so_far):
            # As it may have multiple/consecutive totals or dates in one receipt,
            # the model can predict all of them and the goal
            # is to get the one with the highest probability.
            if new_category in ("total", "date"):
                idx = probabilities.index(new_probability, seps[i], seps[i + 1])
                if 0 <= idx < len(raw_texts):
                    results[new_category] = (raw_texts[idx], str(new_probability))
            else:
                str_list = raw_texts[seps[i]: seps[i + 1]]
                best_predicted_text = " ".join(str_list) if len(str_list) != 0 else ""
                results[new_category] = (best_predicted_text, str(new_probability))
    
    # Then, we clean up date, total and undetected categories.
    null_categories = []
    for category in results.keys():
        best_predicted_text, best_probability = results[category]
        if category == "company":
            results[category] = clean_company(best_predicted_text)
        elif category == "address":
            results[category] = clean_address(best_predicted_text)
        elif category == "date":
            results[category] = extract_date(best_predicted_text)
        elif category == "total":
            results[category] = extract_total(best_predicted_text)
        
        # If the model failed to recognise the category, we store it so that it can be removed later on.
        if len(results[category]) == 0:
            null_categories.append(category)
    
    # Finally, we remove categories with probability of zero or not detected.
    # This will increase the precision while the recall will stay the same.
    for null_category in null_categories:
        results.pop(null_category)
    return results
