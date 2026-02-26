import re

def extract_commands():
    with open("output.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    in_commands = False
    
    current_sheet = ""
    output = []
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
            
        m_sheet = re.match(r"(Activity\s*Sheet\s*\d+)", line, re.IGNORECASE)
        if m_sheet:
            current_sheet = m_sheet.group(1)
            output.append(f"\n=== {current_sheet} ===")
            in_commands = False
            continue
            
        if line == "Operation" and i+1 < len(lines):
            next_line = lines[i+1].strip()
            if "Command" in next_line or "Formula" in next_line:
                in_commands = True
                output.append("\n[Commands]")
                continue
            
        if in_commands and ("Command(s)" in line or "Formula(s)" in line):
            continue

        if line == "Problem Statement" or "Problem Statement" in line:
            in_commands = False
            continue
            
        if line.startswith("Department of Science") or "Department of Science" in line:
            in_commands = False
            continue
            
        if in_commands:
            output.append(line)
            
    with open("commands.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(output))

if __name__ == "__main__":
    extract_commands()
