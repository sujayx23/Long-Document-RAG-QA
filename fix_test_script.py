# Fix the indentation in test_improved_rag.py
with open('test_improved_rag.py', 'r') as f:
    lines = f.readlines()

# Find and fix the problematic section
fixed_lines = []
in_generate_report = False
for line in lines:
    if 'def generate_report' in line:
        in_generate_report = True
    
    # Fix the indentation for the f.write lines
    if in_generate_report and line.strip().startswith('f.write'):
        # Ensure proper indentation (12 spaces for method content)
        fixed_line = ' ' * 12 + line.strip() + '\n'
        fixed_lines.append(fixed_line)
    else:
        fixed_lines.append(line)

# Write back
with open('test_improved_rag.py', 'w') as f:
    f.writelines(fixed_lines)

print("Fixed indentation in test_improved_rag.py")
