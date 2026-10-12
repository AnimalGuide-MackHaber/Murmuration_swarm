with open("app.py", "r") as f:
    lines = f.readlines()

for i in range(793, 867):
    line = lines[i]
    if line.startswith("                "):
        lines[i] = line[8:] # remove 8 spaces

with open("app.py", "w") as f:
    f.writelines(lines)
