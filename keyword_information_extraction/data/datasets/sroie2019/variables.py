# The pattern that will be used to match the date when using regex.
SROIE_DATE_PATTERN = r"(\d{4}|\d{2}|\d)[-/\s.]([A-Za-z]{3}|\d{2}|\d)[-/\s.](\d{4}|\d{2})"

# The pattern that will be used to match the total when using regex.
SROIE_TOTAL_PATTERN = r"^[^\+\-]*(\d+\.(\d{2}|\d))"

# The SROIE vocabulary.
# As the sroie vocabulary has an illegal character '"' at position 2,
# we need to add an escape character at that position.
# This will allow us to use double quotes when you normally would not be allowed
SROIE_VOCAB = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`{|}~·"

# The SROIE text maximum length
SROIE_TEXT_MAX_LENGTH = 68
