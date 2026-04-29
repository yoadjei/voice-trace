# shared khaya api key rotator — alternates between KHAYA_API_KEY and KHAYA_API_KEY_2.
# each key has a 10 req/min limit; alternating them gives ~20 req/min combined.
# with 2 keys: 3s sleep between calls keeps each key at ~10 req/min.
import os
import itertools
from dotenv import load_dotenv

load_dotenv()

KHAYA_RATE_SLEEP = 3.0  # seconds between requests (3s × 2 keys = 6s per key = 10 req/min per key)

_key1 = os.getenv("KHAYA_API_KEY")
_key2 = os.getenv("KHAYA_API_KEY_2")

# build key pool — include key2 only if set
_keys = [k for k in [_key1, _key2] if k]
_key_cycle = itertools.cycle(_keys) if _keys else itertools.cycle([])


def next_key() -> str:
    # returns the next api key in rotation (KHAYA_API_KEY, then KHAYA_API_KEY_2, repeat)
    if not _keys:
        raise RuntimeError(
            "No Khaya API key: set KHAYA_API_KEY in .env (optional: KHAYA_API_KEY_2 to alternate)."
        )
    return next(_key_cycle)


def key_count() -> int:
    return len(_keys)
