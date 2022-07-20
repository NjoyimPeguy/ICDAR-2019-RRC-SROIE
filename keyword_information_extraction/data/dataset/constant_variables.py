# The company pattern not to match.
UNWANTED_COMPANY_PATTERN = r"^(?!.*(TAX|RECEIPT|INVOICE)).*"

# The two date patterns to match.
DATE_PATTERN_1 = r"(\d{4}|\d{2}|\d)[-/.](\d{2}|\d)[-/.](\d{4}|\d{2}|\d)"
DATE_PATTERN_2 = r"(\d{4}|\d{2}|\d)[-/.]*\s*(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[-/.]*\s*(\d{4}|\d{2}|\d)"

# The total pattern to match.
TOTAL_PATTERN = r"(^[^\+\-]|([$\+\-]|[RM\s]))*((\d{3}|\d{2}|\d|\d\,\d{3})\.(\d{2}|\d))"

# As the sroie vocabulary has an illegal character '"' at position 2,
# we need to add an escape character at that position.
# This will allow us to use double quotes when you normally would not be allowed
VOCAB = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`{|}~·"

# The SROIE text maximum length
MAXIMUM_LENGTH = 68
