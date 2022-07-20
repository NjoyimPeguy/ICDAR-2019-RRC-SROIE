import regex
import numpy as np

from typing import List
from collections import OrderedDict
from keyword_information_extraction.utils.misc import is_number
from keyword_information_extraction.data.dataset.constant_variables import DATE_PATTERN_1, DATE_PATTERN_2, \
    TOTAL_PATTERN, UNWANTED_COMPANY_PATTERN


def clean_company(text: str):
    """
    Given a string line, this function will attempt to remove unwanted string that lowers the F1-score.

    Args:
        text (string): The string line.

    Returns:
        A cleaned company.

    """

    # Do not match a company that contains unwanted strings (see pattern).
    unwanted_company = regex.search(UNWANTED_COMPANY_PATTERN, text.strip())
    if unwanted_company is None:
        return ""

    # Three different regex to remove unwanted strings.
    m = regex.match(r"(\d+[^0-9]*[A-Z]+)$", text.strip())
    if m is None:
        m = regex.search(r"\([A-Z]*\d+[^0-9]*[A-Z]+\).*$", text.strip())
        if m is None:
            m = regex.search(r"\([A-Z\s]+[\)]*$", text.strip())
            if m is None:
                return text.strip()

    # Once the unwanted string is found, it is simply removed.
    idx = text.find(m.group().strip())

    text = text[:idx].strip()

    return text


def clean_address(text: str):
    """
    As some receipts have the address and the phone number lying on the same line,
    this function will attempt to remove unwanted phone number that lowers the F1-score.

    Args:
        text (string): The string line to clean to.

    Returns:
        A cleaned address.

    """

    sub_str = "TEL"

    idx = text.find(sub_str)

    if idx != -1:
        text = text[:idx]

    text = regex.sub(r"(\d+[\-][^a-zA-Z]*)$", "", text.strip()).strip()

    return text


def extract_date(text: str):
    """
    Given a string line, this function will extract the date based on a regex.

    Args:
        text (string): The string line for which an attempt to extract a date will be performed.

    Returns:
        A string date based on a regex. Otherwise, empty string.
    """

    # We try to match a date with one regex.
    m = regex.search(DATE_PATTERN_1, text.strip())

    # If the previous regex has failed, we try with another regex
    if m is None:
        m = regex.search(DATE_PATTERN_2, text.strip())

    # If the date has just been found, we return it.
    if m is not None:
        return m.group().strip()

    # Otherwise, an empty string is returned.
    return ""


def extract_total(text: str):
    """
    Given a string line, this function will extract a float number based on a regex.

    Args:
        text (string): The string line for which an attempt to extract a float number will be performed.

    Returns:
        A string float number based on a regex. Otherwise, empty string.
    """

    # We try to match the total with a unique regex
    m = regex.search(TOTAL_PATTERN, text.strip())

    # If the total has just been found, we just return its rounded version.
    if m is not None:
        return m.group().strip()

    # Otherwise, an empty string is returned.
    return ""


def convert_predictions_to_dict(labels_classes: dict,
                                raw_texts: List[str],
                                predicted_classes: List[int],
                                probabilities: List[float]) -> dict:
    """
    Convert the model's predictions into a dictionary.

    Args:
        labels_classes (dict): The dictionary of classes.
        raw_texts (string, list): The raw texts, i.e., texts that are not one-hot encoded.
        predicted_classes (int, list): The model's predicted classes.
        probabilities (float, list):  The model's probabilities.

    Returns:
        A dictionary where keys are the category (company, address, ...) and
        the values are the corresponding raw text associated to a given category.
    """

    # We do not need the "none" class.
    results = {class_name: ("", 0.0) for class_name in labels_classes.keys() if class_name != "none"}

    reversed_entities = {klass: entity for entity, klass in labels_classes.items()}

    # The idea of the list below is to get the span of indices of each entity.
    # For example, the address entity may be situated on multiple text lines which corresponds to index 2 up to index 6.
    seps = [0] + (np.nonzero(np.diff(predicted_classes))[0] + 1).tolist() + [len(predicted_classes)]

    # First, we identify the category predicted by the model.
    for i in range(len(seps) - 1):
        predicted_class = predicted_classes[seps[i]]
        entity = reversed_entities[predicted_class]
        if entity != "none":
            current_max_probability = max(probabilities[seps[i]: seps[i + 1]])
            _, best_probability_so_far = results[entity]
            if current_max_probability > float(best_probability_so_far):
                if entity in ("total", "date"):
                    # As it may have multiple/consecutive totals or dates in one receipt,
                    # the model can predict all of them and
                    # the goal is to get only one of them with the highest probability.
                    idx = probabilities.index(current_max_probability, seps[i], seps[i + 1])
                    if 0 <= idx < len(raw_texts):
                        results[entity] = (raw_texts[idx], str(current_max_probability))
                else:
                    str_list = raw_texts[seps[i]: seps[i + 1]]
                    best_predicted_text = " ".join(str_list) if len(str_list) != 0 else ""
                    results[entity] = (best_predicted_text, str(current_max_probability))

    # Then, we clean up entities.
    null_entities = []
    # Create the text space, i.e., concatenate all the OCR data lines into one string.
    text_space = ""
    for raw_text in raw_texts:
        text_space += raw_text
    text_space = text_space.strip()

    for entity in results.keys():

        best_predicted_text, _ = results[entity]
        if entity == "company":
            results[entity] = clean_company(best_predicted_text)
        elif entity == "address":
            results[entity] = clean_address(best_predicted_text)
        elif entity == "date":
            results[entity] = extract_date(best_predicted_text)
        elif entity == "total":
            results[entity] = extract_total(best_predicted_text)

        # If the model has failed to recognise a given entity,
        # we are going to use based-rules for company, date and total entities.
        if len(results[entity]) == 0:

            # The 'company' rule works as follows:
            # The company name must be in the first two positions if it is not a number,
            # or it does not contain words such as 'TAX', 'RECEIPT' and 'INVOICE'.
            if entity == "company":
                for k in range(2):
                    current_text = raw_texts[k].strip()
                    company_not_to_match = regex.search(UNWANTED_COMPANY_PATTERN, current_text)
                    if company_not_to_match is not None and not is_number(current_text):
                        m = regex.search(r"^\d+", current_text)
                        if m is not None:
                            results[entity] = current_text
                        else:
                            results[entity] = clean_company(current_text)
                        break

            # The 'date' rule is very simple. It is based on the previous regex pattern used to extract a date.
            elif entity == "date":
                for raw_text in raw_texts:
                    date = extract_date(raw_text)
                    if len(date) != 0:
                        results[entity] = date
                        break

            # The 'total' rule works as follows:
            # 1) Save the total if it is only preceded by 'TOTAL', 'AMOUNT' and
            # 2) if it needs some rounding value adjustment.
            elif entity == "total":
                found = False
                for i, raw_text in enumerate(raw_texts):
                    total = extract_total(raw_text.strip())
                    if len(total) != 0:
                        j = i
                        while j > 0 and not found:
                            text = raw_texts[j].strip()
                            line_to_match = regex.search(r"^(.*(TOTAL|AMOUNT)).*", text)
                            line_not_to_match = regex.search(r"^(?!.*(EX|SUB)).*", text)
                            if line_to_match is not None and line_not_to_match is not None:
                                results[entity] = total
                                found = True
                            j -= 1

                    if found:
                        # Try to round the total value.
                        str2Match = regex.search(r"(RND|ROUNDING).*(\d+.\d+)", text_space)
                        if str2Match is not None:
                            rounding_value = regex.search(TOTAL_PATTERN, str2Match.group().strip())
                            if rounding_value is not None:
                                rounding_value = rounding_value.group().strip()
                                if "RM" not in results[entity]:
                                    try:
                                        int_value = float(rounding_value)
                                        total = float(results[entity])
                                        results[entity] = str(total + int_value)
                                    except ValueError:
                                        pass
                        break

            # Eventually, we save the empty entity so that it can be removed later on,
            # if the based-rules above have just failed.
            if len(results[entity]) == 0:
                null_entities.append(entity)

    # Finally, we remove entities with probability of zero or not detected.
    # This will increase the precision while the recall stays the same.
    for null_entity in null_entities:
        results.pop(null_entity)

    return results
