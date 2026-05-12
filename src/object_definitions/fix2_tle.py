def compute_tle_checksum(line):
    total = 0
    for char in line[:68]:
        if char.isdigit():
            total += int(char)
        elif char == '-':
            total += 1
    return total % 10

def fix_tle_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    fixed_lines = []
    for line in lines:
        line = line.rstrip('\n').rstrip()
        if line.startswith('1 ') or line.startswith('2 '):
            content = line[:68].ljust(68)
            checksum = compute_tle_checksum(content)
            line = content + str(checksum)
        fixed_lines.append(line + '\n')

    with open(filepath, 'w') as f:
        f.writelines(fixed_lines)
    print("Done.")

fix_tle_file("/home/thebear/codes4fun/master/sim/GNSS_R_System_Simulator/data/TLE_EPHEMIRIS/CELESTE_IOD.tle")