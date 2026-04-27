import argparse
import pandas as pd

SYMBOL_MAP = {
    '$usdx.csv': '$USDX',
    'usdjpy.csv': 'USDJPY',
    'xauusd.csv': 'XAUUSD',
}

SOURCE_FILES = [
    'data/$usdx.csv',
    'data/usdjpy.csv',
    'data/xauusd.csv',
]

def convert_datetime(date_str, time_str):
    date_str = date_str.replace('.', '-')
    time_str = time_str[:5].replace(':', '-')
    return f"{date_str} {time_str}"

def process_file(filepath):
    with open(filepath, 'r') as f:
        header = f.readline().strip()
    col_names = header.replace('<', '').replace('>', '').split('\t')
    df = pd.read_csv(filepath, sep='\t', skiprows=1, header=None, names=col_names)
    filename = filepath.split('/')[-1].lower()
    symbol = SYMBOL_MAP[filename]
    df['symbol'] = symbol
    df['datetime'] = df.apply(lambda row: convert_datetime(row['DATE'], row['TIME']), axis=1)
    df['tick_volume'] = df['TICKVOL']
    return df[['datetime', 'symbol', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'tick_volume']]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', required=True, help='Output filename without extension')
    args = parser.parse_args()
    dfs = [process_file(f) for f in SOURCE_FILES]
    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.sort_values('datetime', ascending=False)
    merged.columns = merged.columns.str.lower()
    output_path = f"data/{args.n}.csv"
    merged.to_csv(output_path, index=False, encoding='utf-16')
    print(f"Saved to {output_path}")

if __name__ == '__main__':
    main()