import sys

print('Python executable:', sys.executable)
print('Python search paths (sys.path):')
for p in sys.path:
    print(f'- {p}')