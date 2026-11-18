import argparse
import ptsdrift as m


def main():
    ap = argparse.ArgumentParser(description="字幕 PTS 漂移校正")
        ap.add_argument("--anchors", required=True)
        ap.add_argument("--out", required=True)
    a = ap.parse_args()
        print("TODO")


if __name__ == "__main__":
    main()
