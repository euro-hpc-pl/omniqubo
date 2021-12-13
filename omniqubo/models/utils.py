import random
import string

from omniqubo.constants import RAND_STR_LEN


def gen_random_str(n: int = None) -> str:
    if n is None:
        n = RAND_STR_LEN
    str_letters = string.ascii_uppercase + string.digits
    return "".join(random.choices(str_letters, k=n))
