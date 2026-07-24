import sys
from pathlib import Path

if __name__ == "__main__":
    base_dir = Path(sys.argv[1])

    with open(base_dir / "data_schema.txt", "r") as file:
        lines = file.readlines()

    for i in range(0, len(lines), 26):
        with open(base_dir / f"shard_{1+i//26}.txt", "w") as file:
            file.writelines(lines[i:i+26])
