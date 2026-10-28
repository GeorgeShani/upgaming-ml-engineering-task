"""Verify the selected checkpoint's checksum without loading its model."""

import sys

from pedestrian.app_support import read_selection, sha256_file


def main():
    try:
        selection, checkpoint = read_selection()
        actual = sha256_file(checkpoint)
        expected = selection["checkpoint_sha256"]
    except (OSError, ValueError, KeyError) as error:
        print(f"Could not verify checkpoint: {error}", file=sys.stderr)
        return 1

    print(f"Checkpoint: {checkpoint}")
    print(f"SHA-256: {actual}")
    if actual != expected:
        print("Checksum does not match the selected model.", file=sys.stderr)
        return 1
    print("Checksum matches the selected model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
