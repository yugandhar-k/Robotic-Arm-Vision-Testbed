"""search_and_pick_eye_in_hand.py - Search-and-pick, pinned to the eye-in-hand camera.

A dedicated entry point for one search view, so there is nothing to switch and no `V`
key to worry about - just run this file. Identical behaviour to
`python search_and_pick.py --view eye_in_hand`; see search_and_pick.py's module
docstring for how the eye-in-hand search sweep, confirm, grasp and place pipeline works.

Usage
-----
    python search_and_pick_eye_in_hand.py
    python search_and_pick_eye_in_hand.py --seed 7

Author: Yugandhar Kulkarni - DRDO Research Intern - DES Pune University
"""
import argparse

from search_and_pick import run_demo
from src.utils import RANDOM_SEED


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED,
                        help="object layout seed")
    args = parser.parse_args()
    run_demo(args.seed, "eye_in_hand")


if __name__ == "__main__":
    main()
