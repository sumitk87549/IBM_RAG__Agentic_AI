import json

notebook_path = "Fundamentals of Building AI Agents/AI-Math-Assistant Tool Calling.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        source = cell.get('source', [])
        new_source = []
        for line in source:
            if 'from langchain.agents import AgentType' in line:
                # Remove this line or comment it out
                pass
            elif 'AgentType.ZERO_SHOT_REACT_DESCRIPTION' in line:
                line = line.replace('AgentType.ZERO_SHOT_REACT_DESCRIPTION', '"zero-shot-react-description"')
                new_source.append(line)
            else:
                new_source.append(line)
        cell['source'] = new_source
        
    elif cell.get('cell_type') == 'markdown':
        source = cell.get('source', [])
        new_source = []
        for line in source:
            if 'AgentType.ZERO_SHOT_REACT_DESCRIPTION' in line:
                line = line.replace('AgentType.ZERO_SHOT_REACT_DESCRIPTION', '"zero-shot-react-description"')
                new_source.append(line)
            else:
                new_source.append(line)
        cell['source'] = new_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
    
print("Notebook AgentType fixed successfully.")
