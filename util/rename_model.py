import argparse
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Rename model folder prefix and update live.mq5')
    parser.add_argument('full_old_name', help='Full old model folder name (e.g., mymodel-3945-454534)')
    parser.add_argument('new_prefix', help='New prefix for the model folder')
    args = parser.parse_args()

    models_dir = Path('models')
    old_name = args.full_old_name
    new_prefix = args.new_prefix

    old_path = models_dir / old_name
    if not old_path.exists():
        print(f"Error: Model folder not found: {old_path}")
        return

    suffix = old_name.split('-', 1)[1] if '-' in old_name else ''
    new_name = f"{new_prefix}-{suffix}" if suffix else new_prefix
    new_path = models_dir / new_name

    old_path.rename(new_path)
    print(f"Renamed: {old_name} -> {new_name}")

    live_mq5 = Path('live.mq5')
    if not live_mq5.exists():
        print("live.mq5 not found, skipping update")
        return

    content = live_mq5.read_text()
    updated = content

    content = content.replace(f'models/{old_name}', f'models/{new_name}')
    content = content.replace(f'models\\\\{old_name}', f'models\\\\{new_name}')

    if content != updated:
        live_mq5.write_text(content)
        print(f"Updated live.mq5 to use model: {new_name}")
    else:
        print("live.mq5 doesn't reference old model, no update needed")


if __name__ == '__main__':
    main()