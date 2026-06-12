"""
Camera Index Checker

Iterates through available camera indices and shows a live preview
for each working camera.  Useful for finding which video index
corresponds to which physical USB camera.

Usage::

    python -m tools.camera_checker
    python -m tools.camera_checker --start 0 --end 9
    python -m tools.camera_checker --start 4 --end 4  # single camera
"""

from __future__ import annotations

import argparse
import sys

import cv2

_KEY_NEXT: int = ord("n")
_KEY_QUIT: int = ord("q")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Iterate camera indices to find working cameras.",
    )
    parser.add_argument(
        "--start",
        "-s",
        type=int,
        default=0,
        help="First camera index to check (default: 0).",
    )
    parser.add_argument(
        "--end",
        "-e",
        type=int,
        default=9,
        help="Last camera index to check (inclusive, default: 9).",
    )
    parser.add_argument(
        "--width",
        "-w",
        type=int,
        default=640,
        help="Preview width (default: 640).",
    )
    parser.add_argument(
        "--height",
        "-H",
        type=int,
        default=480,
        help="Preview height (default: 480).",
    )
    return parser.parse_args()


def check_cameras(
    start: int = 0,
    end: int = 9,
    width: int = 640,
    height: int = 480,
) -> None:
    """Scan camera indices and show live preview for each working one."""
    print("=" * 50)
    print("   CAMERA INDEX CHECKER")
    print("=" * 50)
    print(f"   Range: {start} – {end}")
    print(f"   Press  [n]  next camera")
    print(f"   Press  [q]  quit")
    print("=" * 50)

    for idx in range(start, end + 1):
        print(f"\n[INFO] Opening camera index {idx} ...", end=" ")
        cap = cv2.VideoCapture(idx)

        if not cap.isOpened():
            print("NOT FOUND")
            continue

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        print("OK")
        print("       Press [n] → next   [q] → quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Frame read failed — skipping.")
                break

            cv2.putText(
                frame,
                f"CAMERA INDEX: {idx}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Camera Checker", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == _KEY_NEXT:
                break
            if key == _KEY_QUIT:
                cap.release()
                cv2.destroyAllWindows()
                print("\nDone.")
                return

        cap.release()
        cv2.destroyAllWindows()

    print("\nAll indices checked.")


def main() -> None:
    args = _parse_args()
    check_cameras(
        start=args.start,
        end=args.end,
        width=args.width,
        height=args.height,
    )


if __name__ == "__main__":
    main()
