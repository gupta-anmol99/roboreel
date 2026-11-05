# Parse environment.yml without PyYAML (no external deps needed)
# Splits into conda-only yml and a pip requirements.txt

with open('/tmp/environment.yml') as f:
    lines = f.readlines()

in_pip_section = False
conda_lines = []
pip_deps = []

for line in lines:
    stripped = line.rstrip()
    if stripped == '  - pip:':
        in_pip_section = True
        continue
    if in_pip_section:
        if stripped.startswith('      - '):   # pip items are 6-space indented
            pip_deps.append(stripped[8:].strip())
        else:
            in_pip_section = False            # anything else ends the pip block
            conda_lines.append(line)
    else:
        conda_lines.append(line)

with open('/tmp/conda_only.yml', 'w') as f:
    f.writelines(conda_lines)

with open('/tmp/pip_requirements.txt', 'w') as f:
    f.write('\n'.join(pip_deps) + '\n')

print(f"Conda packages: {sum(1 for l in conda_lines if l.strip().startswith('- '))}")
print(f"Pip packages:   {len(pip_deps)}")
