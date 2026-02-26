import re

def extract_viva_questions():
    with open("output.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    in_viva = False
    in_tasks = False
    
    current_sheet = ""
    output = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        m_sheet = re.match(r"(Activity\s*Sheet\s*\d+)", line, re.IGNORECASE)
        if m_sheet:
            current_sheet = m_sheet.group(1)
            output.append(f"\n=== {current_sheet} ===")
            in_viva = False
            in_tasks = False
            continue
            
        if line == "Tasks":
            in_tasks = True
            in_viva = False
            output.append("\n[Tasks]")
            continue
            
        if line == "Results and Inference" or line.startswith("Results and"):
            in_tasks = False
            in_viva = False
            continue
            
        if line == "Viva Questions":
            in_viva = True
            in_tasks = False
            output.append("\n[Viva Questions]")
            continue
            
        if line.startswith("Department of Science"):
            in_viva = False
            in_tasks = False
            continue
            
        if in_tasks or in_viva:
            output.append(line)
            
    with open("questions.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(output))

if __name__ == "__main__":
    extract_viva_questions()
