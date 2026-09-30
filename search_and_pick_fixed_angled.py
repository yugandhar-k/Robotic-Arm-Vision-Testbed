"""search_and_pick_fixed_angled.py - Search-and-pick, pinned to the fixed-angled (45deg)
camera.

A dedicated entry point for one search view, so there is nothing to switch and no `V`
key to worry about - just run this file. The fixed-angled camera does everything: finds
the object, localises it, and plans the grasp - the eye-in-hand camera is never used, not
even to confirm before grasping (see search_and_pick.py's `SearchAndPlaceController`,
specifically how "confirm_move" is handled for a non-eye_in_hand search view). Identical
behaviour to `python search_and_pick.py --view fixed_angled`.

Usage
-----
    python search_and_pick_fixed_angled.py
    python search_and_pick_fixed_angled.py --seed 7

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
    run_demo(args.seed, "fixed_angled")


if __name__ == "__main__":
    main()
