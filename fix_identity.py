"""Replace the identity directive in answerer.py"""
import re

with open(r"c:\Project\ASKPERA\PERAASK\answerer.py", "r", encoding="utf-8") as f:
    content = f.read()

# Find and replace line containing Identity directive
old_pattern = r'.*\*\*Identity\*\*.*Ahsan.*Lead AI.*\n'
new_line = '        "11. **Identity (Developer ONLY)**: ONLY if asked who CREATED/BUILT/DEVELOPED this AI (words: creator, developer, banaya, father, Ahsan), say it was developed by **Muhammad Ahsan Sajjad**. He is NOT a PERA employee. Do NOT confuse him with CTO or any PERA role. For PERA role names, answer ONLY from Context.\\n"\n'

match = re.search(old_pattern, content)
if match:
    content = content[:match.start()] + new_line + content[match.end():]
    with open(r"c:\Project\ASKPERA\PERAASK\answerer.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("SUCCESS: Identity directive replaced")
else:
    print("NOT FOUND")
    for i, line in enumerate(content.split('\n')):
        if 'Identity' in line and 'Ahsan' in line:
            print(f"Line {i+1}: {line[:150]}")
