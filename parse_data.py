import polars as pl
from pathlib import Path

def load_window(file: Path) -> pl.DataFrame:
    df = pl.read_ipc(file)
    return df


if __name__ == "__main__":
    import sys
    
    file = Path(sys.argv[1])
    df = load_window(file)

    print(df)
