#!/usr/bin/env python3
"""
test_consensus_logic.py

Standalone test of stair_consensus.py's decision logic - no ROS2, no Docker,
no rclpy import at all. Just the same agree-count math, run against every
possible combination of votes, so you can confirm the state machine itself
is correct before touching any ROS2 wiring.
"""


def decide_mode(fusion_detected: bool, yolo_detected: bool) -> str:
    """Exact same logic as StairConsensus.publish_result()'s mode decision."""
    agree_count = int(fusion_detected) + int(yolo_detected)

    if agree_count == 2:
        return 'STAIR_MODE'
    elif agree_count == 1:
        return 'APPROACH_MODE'
    else:
        return 'NONE'


def main():
    cases = [
        (False, False, 'NONE'),
        (True,  False, 'APPROACH_MODE'),
        (False, True,  'APPROACH_MODE'),
        (True,  True,  'STAIR_MODE'),
    ]

    print(f"{'fusion':<8} {'yolo':<8} {'expected':<15} {'got':<15} {'result'}")
    print('-' * 60)

    all_passed = True
    for fusion, yolo, expected in cases:
        got = decide_mode(fusion, yolo)
        passed = got == expected
        all_passed &= passed
        status = 'PASS' if passed else 'FAIL'
        print(f"{str(fusion):<8} {str(yolo):<8} {expected:<15} {got:<15} {status}")

    print('-' * 60)
    print('ALL PASSED' if all_passed else 'SOME FAILED')


if __name__ == '__main__':
    main()