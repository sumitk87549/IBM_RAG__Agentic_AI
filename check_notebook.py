import json
notebook_path = "Fundamentals of Building AI Agents/AI-Math-Assistant Tool Calling.ipynb"
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)
found = False
for cell in nb.get('cells', []):
    source = "".join(cell.get('source', []))
    if 'AgentType' in source:
        print("Found AgentType in cell:", source)
        found = True
if not found:
    print("AgentType NOT found in notebook.")
