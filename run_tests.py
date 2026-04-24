import os
import subprocess

for b in ['tensorflow', 'torch', 'jax']:
    env = {**os.environ, 'KERAS_BACKEND': b}
    r = subprocess.run(['pytest', f'tests/test_tkan.py', '-k', b], env=env, capture_output=True, text=True)
    print(f"\n--- {b.upper()} ---\n{r.stdout}")
    if r.returncode:
        print(r.stderr)
        exit(1)
print("\nAll passed")
