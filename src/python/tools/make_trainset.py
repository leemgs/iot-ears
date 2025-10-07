
import os
import argparse

def main():
    ap = argparse.ArgumentParser(description='Generate TrainSet.txt by scanning <dataset_dir>/<fine_label>/*.wav')
    ap.add_argument('--dataset_dir', required=True, help='Root directory containing per-label subfolders of wav files')
    ap.add_argument('--out', default='examples/Dataset/TrainSet.txt', help='Output TrainSet.txt path (relative or absolute)')
    ap.add_argument('--rel_base', default=None, help='If set, paths in TrainSet.txt will be relative to this base directory')
    args = ap.parse_args()

    lines = []
    for label in sorted(os.listdir(args.dataset_dir)):
        lpath = os.path.join(args.dataset_dir, label)
        if not os.path.isdir(lpath):
            continue
        for fname in sorted(os.listdir(lpath)):
            if not fname.lower().endswith('.wav'):
                continue
            full = os.path.join(lpath, fname)
            if args.rel_base:
                try:
                    rel = os.path.relpath(full, args.rel_base)
                except ValueError:
                    rel = full
            else:
                rel = os.path.relpath(full, args.dataset_dir)
            lines.append(f"{rel} {label}")

    if not lines:
        raise SystemExit("No .wav files found. Check folder structure: <dataset_dir>/<fine_label>/*.wav")

    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote {len(lines)} entries → {args.out}")

if __name__ == '__main__':
    main()
