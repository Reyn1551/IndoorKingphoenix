"""
HSV Colour Tuner

Opens a camera feed and provides trackbars to tune HSV lower/upper
bounds for colour detection.  Shows the original frame, the mask,
and the masked result side-by-side.  Useful for finding the right
HSV range for gate detection (red / yellow).

Usage::

    python -m tools.hsv_tuner
    python -m tools.hsv_tuner --index 6
    python -m tools.hsv_tuner --index 4 --width 320
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune HSV colour ranges with live trackbars.",
    )
    parser.add_argument(
        "--index",
        "-i",
        type=int,
        default=0,
        help="Camera index (default: 0).",
    )
    parser.add_argument(
        "--width",
        "-w",
        type=int,
        default=320,
        help="Camera width in pixels (default: 320).",
    )
    parser.add_argument(
        "--height",
        "-H",
        type=int,
        default=240,
        help="Camera height in pixels (default: 240).",
    )
    return parser.parse_args()


def _nothing(_: int) -> None:
    """Trackbar callback — no-op."""


def hsv_tuner(cam_index: int, width: int, height: int) -> None:
    """Open *cam_index*, show HSV trackbars, print values on quit."""

    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        print(f"ERROR: Camera {cam_index} could not be opened.")
        return

    win_name = "HSV Tuner"
    cv2.namedWindow(win_name)

    # Trackbars
    cv2.createTrackbar("H Min", win_name, 0, 179, _nothing)
    cv2.createTrackbar("S Min", win_name, 0, 255, _nothing)
    cv2.createTrackbar("V Min", win_name, 0, 255, _nothing)
    cv2.createTrackbar("H Max", win_name, 179, 179, _nothing)
    cv2.createTrackbar("S Max", win_name, 255, 255, _nothing)
    cv2.createTrackbar("V Max", win_name, 255, 255, _nothing)

    # Sensible defaults (yellow-ish)
    cv2.setTrackbarPos("S Min", win_name, 100)
    cv2.setTrackbarPos("V Min", win_name, 100)

    print("\nAdjust sliders until the object is white and background is black.")
    print("Press  [q]  to quit.\n")

    # Initialise so they're always bound even if the loop breaks early
    lower = np.array([0, 0, 0], dtype=np.uint8)
    upper = np.array([179, 255, 255], dtype=np.uint8)
    max_area = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        blur = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        lower = np.array(
            [
                cv2.getTrackbarPos("H Min", win_name),
                cv2.getTrackbarPos("S Min", win_name),
                cv2.getTrackbarPos("V Min", win_name),
            ]
        )
        upper = np.array(
            [
                cv2.getTrackbarPos("H Max", win_name),
                cv2.getTrackbarPos("S Max", win_name),
                cv2.getTrackbarPos("V Max", win_name),
            ]
        )

        mask = cv2.inRange(hsv, lower, upper)

        # Largest-contour area for reference
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        max_area = 0
        if contours:
            largest = max(contours, key=cv2.contourArea)
            max_area = cv2.contourArea(largest)
            cv2.drawContours(frame, [largest], -1, (0, 255, 0), 2)

        result = cv2.bitwise_and(frame, frame, mask=mask)

        # Stack: original | mask | result
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.putText(
            mask_bgr,
            f"AREA: {int(max_area)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        stacked = np.hstack([frame, mask_bgr, result])
        cv2.imshow(win_name, stacked)

        # Live-print for easy copy
        print(
            f"LOWER: [{lower[0]}, {lower[1]}, {lower[2]}]  "
            f"UPPER: [{upper[0]}, {upper[1]}, {upper[2]}]  "
            f"AREA: {int(max_area):>6}   ",
            end="\r",
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Final values
    print("\n\n=== FINAL VALUES (copy these) ===")
    print(f"lower = [{lower[0]}, {lower[1]}, {lower[2]}]")
    print(f"upper = [{upper[0]}, {upper[1]}, {upper[2]}]")
    print(f"(Use min_gate_area < {int(max_area)} for contour filtering)")


def main() -> None:
    args = _parse_args()
    hsv_tuner(args.index, args.width, args.height)


if __name__ == "__main__":
    main()
