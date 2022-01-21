import random
import string

from omniqubo.constants import RAND_STR_LEN


def gen_random_str(n: int = None) -> str:
    """Generates random string of length n

    The string consist of uppercase letters and digits.

    :param n: length of the string, defaults to RAND_STR_LEN
    :return: random string
    """
    if n is None:
        n = RAND_STR_LEN
    str_letters = string.ascii_uppercase + string.digits
    return "".join(random.choices(str_letters, k=n))
