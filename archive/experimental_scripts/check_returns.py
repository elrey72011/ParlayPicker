import ast

filename = 'app_core/apisports.py'

with open(filename, 'r') as f:
    tree = ast.parse(f.read())

for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef):
        returns_collection = False
        if node.returns:
            # Check if return annotation suggests a collection (List, Dict, Sequence, etc.)
            # This is a heuristic.
            s = ast.unparse(node.returns)
            if 'List' in s or 'Dict' in s or 'Sequence' in s:
                returns_collection = True

        if returns_collection:
            for child in ast.walk(node):
                if isinstance(child, ast.Return):
                    if child.value is None or (isinstance(child.value, ast.Constant) and child.value.value is None):
                        print(f"Function '{node.name}' (Line {node.lineno}) returns None but annotated as {s}")
