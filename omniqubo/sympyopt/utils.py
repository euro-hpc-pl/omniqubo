import random
import string

RAND_STR_LEN = 16


def gen_random_str(n: int = None) -> str:
    if n is None:
        n = RAND_STR_LEN
    str_letters = string.ascii_uppercase + string.digits
    return "".join(random.choices(str_letters, k=n))
