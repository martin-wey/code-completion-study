from .markers import SPECIAL_MARKERS


def print_node(node, depth=0):
    pos_point = f"[{node.start_point},{node.end_point}]"
    pos_byte = f"({node.start_byte},{node.end_byte})"
    tab = depth * '\t'
    print(
        f"{tab}"
        f"{repr(node.type):<25}{'is_named' if node.is_named else '-':<20}"
        f"{pos_point:<30}{pos_byte}"
    )


def walk_java(tree, data):
    root = tree.root_node
    code_flow = []

    def _traverse_decl(node, depth=0):
        if depth == 0 and len(code_flow) == 0:
            code_flow.append(SPECIAL_MARKERS['function'])
        if node.type == 'formal_parameter':
            code_flow.append(SPECIAL_MARKERS['parameter'])
        elif node.type == 'identifier':
            code_flow.append(data[node.start_byte:node.end_byte])

        for child in node.children:
            _traverse_decl(child, depth + 1)

    def _traverse_block(node, depth=0):
        if node.type == 'method_invocation':
            code_flow.append(SPECIAL_MARKERS['call'])

        elif node.type == 'argument_list' and \
                len(node.children) > 2:
            code_flow.append(SPECIAL_MARKERS['arguments'])

        elif node.type == 'local_variable_declaration' or node.type == 'assignment_expression':
            code_flow.append(SPECIAL_MARKERS['variable'])

        elif node.type == '=':
            code_flow.append(SPECIAL_MARKERS['value'])

        elif node.type == 'identifier' or node.type == 'type_identifier':
            if node.parent.type != 'method_invocation' and node.parent.type != 'argument_list':
                code_flow.append(SPECIAL_MARKERS['identifier'])
            code_flow.append(data[node.start_byte:node.end_byte])

        next_child = None
        for child in node.children:
            if node.type == 'method_invocation' and child.type == 'method_invocation':
                next_child = child
            else:
                _traverse_block(child, depth + 1)
        if next_child is not None:
            _traverse_block(next_child, depth + 1)

    for child in root.children:
        # first retrieve the function declaration
        if child.type == 'local_variable_declaration' or child.type == 'ERROR':
            _traverse_decl(child)
        # parse the function block
        elif child.type == 'block' or child.type == 'expression_statement':
            _traverse_block(child)

    return code_flow


def walk_c_sharp(tree, data):
    root = tree.root_node
    code_flow = []
    super_block_types = ['block', 'argument_list', 'invocation_expression', 'variable_declaration']

    def _traverse(node, depth=0):
        print_node(node, depth)

        for child in node.children:
            _traverse(child, depth + 1)

    def _traverse_decl(node, depth=0):
        if depth == 0 and len(code_flow) == 0:
            code_flow.append(SPECIAL_MARKERS['function'])
        if node.type == 'parameter':
            code_flow.append(SPECIAL_MARKERS['parameter'])
        elif node.type == 'identifier' and (node.parent.type == 'variable_declaration' or len(code_flow) == 1):
            code_flow.append(data[node.start_byte:node.end_byte])
        elif node.type == 'identifier' and node.parent.type == 'local_function_statement':
            if code_flow[-1] != SPECIAL_MARKERS['function']:
                code_flow[-1] = data[node.start_byte:node.end_byte]
            else:
                code_flow.append(data[node.start_byte:node.end_byte])
        elif node.type == 'identifier' and node.parent.type == 'parameter':
            if code_flow[-1] != SPECIAL_MARKERS['parameter']:
                code_flow[-1] = data[node.start_byte:node.end_byte]
            else:
                code_flow.append(data[node.start_byte:node.end_byte])
        elif node.type == 'ERROR' and node.parent.type == 'tuple_pattern':
            code_flow.append(SPECIAL_MARKERS['parameter'])
            code_flow.append(data[node.start_byte:node.end_byte])

        for child in node.children:
            if child.type != 'block':
                _traverse_decl(child, depth + 1)
            else:
                _traverse_block(child, None, depth + 1)

    def _traverse_block(node, last_block_type=None, depth=0):

        if node.type == 'identifier':
            if last_block_type != 'invocation_expression' and last_block_type != 'argument_list':
                code_flow.append(SPECIAL_MARKERS['identifier'])
            elif last_block_type == 'invocation_expression':
                code_flow.append(SPECIAL_MARKERS['call'])
            elif node.parent.type == 'argument':
                code_flow.append(SPECIAL_MARKERS['arguments'])
            code_flow.append(data[node.start_byte:node.end_byte])

        elif node.type == 'variable_declaration' or node.type == 'assignment_expression':
            code_flow.append(SPECIAL_MARKERS['variable'])

        elif node.type == '=':
            code_flow.append(SPECIAL_MARKERS['value'])

        for child in node.children:
            if child.type in super_block_types:
                _traverse_block(child, child.type, depth + 1)
            else:
                _traverse_block(child, last_block_type, depth + 1)

    for child in root.children:
        _traverse_decl(child)

    return code_flow
