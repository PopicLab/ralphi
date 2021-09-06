import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-process frag panel output")
    parser.add_argument('--in_table', help='Input file')
    parser.add_argument('--out_panel', help='Output file')
    args = parser.parse_args()

    fragment_files = []
    with open(args.in_table, 'r') as f:
        for frag_line in f:
            fragment_files = [x.strip().replace("\"", "") for x in frag_line[1:-2].split(",")]

    with open(args.out_panel, mode='wt', encoding='utf-8') as output:
        output.write('\n'.join(fragment_files))
