import os
import io
import datetime
import javalang
import math
import networkx as nx

METRICS = set(['xmet', 'vref', 'vdec', 'nos', 'nopr', 'noa',
               'nexp', 'nand', 'mdn', 'loop', 'lmet', 'hvoc',
               'heff', 'hdif', 'exct', 'excr', 'cref', 'comp',
               'cast', 'nbltrl', 'ncltrl', 'nsltrl', 'nnlstrl', 'nnulltrl',
               'mnp', 'nfci', 'ndi'])

def tokenize(file_path):
    file = io.open(file_path, 'r', encoding='utf-8')
    try:
        tokens = list(javalang.tokenizer.tokenize(file.read()))
    except:
        with open('./failed.txt', 'a+') as failed_file:
            failed_file.write(datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S')+ ':' + file_path + '\n')
            failed_file.close
            return
    return tokens

def parsed_mem(file_path):
    file = io.open(file_path, 'r', encoding='utf-8')
    try:
        tokens = javalang.tokenizer.tokenize(file.read())
        parser = javalang.parser.Parser(tokens)
        parsed_member = parser.parse_member_declaration()
    except:
        with open('./failed.txt', 'a+') as failed_file:
            failed_file.write(datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S')+ ':' + file_path + '\n')
            failed_file.close
            return
    return parsed_member

def getxmet(parsed_member):
    external_method_count = 0
    for path, node in parsed_member:
        if isinstance(node, javalang.tree.MethodInvocation):
            if node.qualifier is not None:
                external_method_count += 1

    return external_method_count

def getvref(parsed_member):
    variable_references_count = 0

    function_parameters = set()

    for path, node in parsed_member:

        if isinstance(node, javalang.tree.FormalParameter):
            function_parameters.add(node.name)
            variable_references_count += 1



        elif isinstance(node, javalang.tree.MemberReference):
            variable_references_count += 1



        elif isinstance(node, javalang.tree.VariableDeclarator):
            variable_references_count += 1


    return variable_references_count

def getvdec(parsed_member):
    local_variable_declarations_count = 0

    for path, node in parsed_member:
        if isinstance(node, javalang.tree.VariableDeclarator):
            local_variable_declarations_count += 1

    return local_variable_declarations_count

def getnos(parsed_member):
    statement_count = 0

    for path, node in parsed_member:
        if isinstance(node, javalang.tree.Statement):
            statement_count += 1

    return statement_count

def getnopr(parsed_member):
    operator_count = 0

    for path, node in parsed_member:
        if isinstance(node, javalang.tree.Assignment):
            operator_count += 1


        elif isinstance(node, javalang.tree.BinaryOperation):
            operator_count += 1



        elif isinstance(node, javalang.tree.TernaryExpression):
            operator_count += 1



        elif isinstance(node, javalang.tree.MemberReference):
            if node.postfix_operators:
                for operator in node.postfix_operators:
                    operator_count += 1


        elif isinstance(node, javalang.tree.VariableDeclarator):
            if node.initializer is not None:
                operator_count += 1  # 这里的赋值使用 '='


        elif isinstance(node, javalang.tree.BinaryOperation) and node.operator == 'instanceof':
            operator_count += 1

    return operator_count

def getnoa(parsed_member):
    parameter_count = 0
    method_invocation_parameter_count = 0

    for path, node in parsed_member:
        if isinstance(node, javalang.tree.FormalParameter):
            parameter_count += 1


        elif isinstance(node, javalang.tree.MethodInvocation):
            method_invocation_parameter_count += len(node.arguments)


    sum_cnt = parameter_count + method_invocation_parameter_count

    return sum_cnt

def getnexp(parsed_member):
    expression_count = 0


    for path, node in parsed_member:

        if isinstance(node, javalang.tree.Assignment):
            expression_count += 1

        elif isinstance(node, javalang.tree.MethodInvocation):
            expression_count += 1

        elif isinstance(node, javalang.tree.BinaryOperation):
            expression_count += 1

        elif isinstance(node, javalang.tree.Literal):
            expression_count += 1

        elif isinstance(node, javalang.tree.TernaryExpression):
            expression_count += 1

        elif isinstance(node, javalang.tree.MemberReference):
            expression_count += 1

    return expression_count

def getnand(parsed_member):
    operand_count = 0


    for path, node in parsed_member:

        if isinstance(node, javalang.tree.MemberReference):
            operand_count += 1

        elif isinstance(node, javalang.tree.VariableDeclarator):
            operand_count += 1

        elif isinstance(node, javalang.tree.Literal):
            operand_count += 1

        elif isinstance(node, javalang.tree.MethodInvocation):
            operand_count += 1

    return operand_count

def getlmet(parsed_member):
    local_method_count = 0

    for path, node in parsed_member:
        if isinstance(node, javalang.tree.MethodInvocation):
            if node.qualifier is None or node.qualifier == 'this':
                local_method_count += 1

    return local_method_count

def getexct(parsed_member):
    thrown_exceptions_count = 0
    declared_exceptions_count = 0

    for path, node in parsed_member:
        if isinstance(node, javalang.tree.ThrowStatement):
            thrown_exceptions_count += 1

        if isinstance(node, (javalang.tree.MethodDeclaration, javalang.tree.ConstructorDeclaration)):
            if node.throws:
                declared_exceptions_count += len(node.throws)

    return thrown_exceptions_count + declared_exceptions_count

def getexcr(parsed_member):
    exception_references_count = 0


    for path, node in parsed_member:

        if isinstance(node, javalang.tree.CatchClause):
            exception_references_count += 1

        if isinstance(node, (javalang.tree.MethodDeclaration, javalang.tree.ConstructorDeclaration)):
            if node.throws:
                exception_references_count += len(node.throws)

    return exception_references_count

def getcref(parsed_member):

    referenced_classes = set()


    for path, node in parsed_member:

        if isinstance(node, javalang.tree.FormalParameter):
            referenced_classes.add(node.type.name)

        elif isinstance(node, javalang.tree.VariableDeclaration):
            if isinstance(node.type, javalang.tree.ReferenceType):
                referenced_classes.add(node.type.name)

    return len(referenced_classes)

def getcomp(parsed_member):

    cyclomatic_complexity = 1

    for path, node in parsed_member:

        if isinstance(node, javalang.tree.IfStatement):
            cyclomatic_complexity += 1

        elif isinstance(node, (javalang.tree.ForStatement, javalang.tree.WhileStatement, javalang.tree.DoStatement)):
            cyclomatic_complexity += 1

        elif isinstance(node, javalang.tree.CatchClause):
            cyclomatic_complexity += 1

        elif isinstance(node, javalang.tree.SwitchStatement):

            cyclomatic_complexity += len(node.cases)

        elif isinstance(node, javalang.tree.BinaryOperation):
            if node.operator in ['&&', '||']:
                cyclomatic_complexity += 1

    return cyclomatic_complexity

def getcast(parsed_member):

    type_cast_count = 0

    for path, node in parsed_member:

        if isinstance(node, javalang.tree.Cast):
            type_cast_count += 1

    return type_cast_count

def getnbltrl(parsed_member):

    boolean_literal_count = 0


    for path, node in parsed_member:

        if isinstance(node, javalang.tree.Literal):
            if node.value == 'true' or node.value == 'false':
                boolean_literal_count += 1

    return boolean_literal_count

def getncltrl(parsed_member):

    char_literal_count = 0

    for path, node in parsed_member:

        if isinstance(node, javalang.tree.Literal):

            if isinstance(node.value, str) and len(node.value) == 3 and node.value.startswith(
                    "'") and node.value.endswith("'"):
                char_literal_count += 1

    return char_literal_count

def getnsltrl(parsed_member):

    string_literal_count = 0

    for path, node in parsed_member:
        if isinstance(node, javalang.tree.Literal):

            if isinstance(node.value, str) and node.value.startswith('"') and node.value.endswith('"'):
                string_literal_count += 1

    return string_literal_count

def getnnltrl(parsed_member):

    numeric_literal_count = 0


    for path, node in parsed_member:

        if isinstance(node, javalang.tree.Literal):

            if node.value.replace('.', '', 1).isdigit() or (
                    node.value[:-1].isdigit() and node.value[-1] in ['L', 'l', 'F', 'f', 'D', 'd']):
                numeric_literal_count += 1

    return numeric_literal_count

def getnnulltrl(parsed_member):

    null_literal_count = 0

    for path, node in parsed_member:

        if isinstance(node, javalang.tree.Literal):

            if node.value == 'null':
                null_literal_count += 1

    return null_literal_count

def gethvoc(parsed_member):

    operators = set()
    operands = set()

    for path, node in parsed_member:

        if isinstance(node, javalang.tree.Assignment):
            operators.add(node.type)
        elif isinstance(node, javalang.tree.BinaryOperation):
            operators.add(node.operator)
        elif isinstance(node, javalang.tree.TernaryExpression):
            operators.add('? :')

        elif isinstance(node, javalang.tree.MemberReference):
            operands.add(node.member)
            if node.postfix_operators:
                for operator in node.postfix_operators:
                    if operator == '++' or operator == '--':
                        operators.add(operator)


        elif isinstance(node, javalang.tree.VariableDeclarator):
            if node.initializer is not None:
                operators.add("=")
                operands.add(node.name)


        elif isinstance(node, javalang.tree.MemberReference):
            operands.add(node.member)


        elif isinstance(node, javalang.tree.Literal):
            operands.add(node.value)


        elif isinstance(node, javalang.tree.MethodInvocation):
            operands.add(node.member)


    halstead_vocabulary = len(operators) + len(operands)

    return halstead_vocabulary

def gethdif(parsed_member):
    operators = set()
    operands = set()

    N2 = getnand(parsed_member)

    for path, node in parsed_member:
        if isinstance(node, javalang.tree.Assignment):
            operators.add(node.type)

        elif isinstance(node, javalang.tree.BinaryOperation):
            operators.add(node.operator)

        elif isinstance(node, javalang.tree.TernaryExpression):
            operators.add('? :')


        elif isinstance(node, javalang.tree.MemberReference):
            operands.add(node.member)

            if node.postfix_operators:
                for operator in node.postfix_operators:
                    if operator == '++' or operator == '--':
                        operators.add(operator)

        elif isinstance(node, javalang.tree.VariableDeclarator):
            if node.initializer is not None:
                operators.add("=")
                operands.add(node.name)


        elif isinstance(node, javalang.tree.MemberReference):
            operands.add(node.member)

        elif isinstance(node, javalang.tree.Literal):
            operands.add(node.value)

        elif isinstance(node, javalang.tree.MethodInvocation):
            operands.add(node.member)


    n1 = len(operators)
    n2 = len(operands)

    if n2 > 0:
        D = (n1 / 2) * (N2 / n2)
    else:
        D = 0

    return round(D, 2)

def getheff(parsed_member):
    hdif = gethdif(parsed_member)
    hvoc = gethvoc(parsed_member)

    N1 = getnopr(parsed_member)
    N2 = getnand(parsed_member)

    N = N1 + N2


    if hvoc > 0:
        V = N * math.log2(hvoc)
    else:
        V = 0


    E = hdif * V

    return round(E, 2)

def getmdn(tokens):
    max = 0
    depth = 0
    for line in tokens:
        if line.value == '{':
            depth += 1
        elif line.value == '}':
            max = max if max > depth else depth
            depth -= 1
    return max

def getloop(tokens):
    loop_num = 0
    for line in tokens:
        if line.value == 'for':
            loop_num += 1
        elif line.value == 'while':
            loop_num += 1
    return loop_num

def getmnp(tokens):
    parallel_num = 0
    depth = 0
    for line in tokens:
        if line.value == '{':
            depth += 1
        elif line.value == '}':
            depth -= 1
            if depth == 1:
                parallel_num += 1
    return parallel_num
def getnfci(tokens):
    if_num = 0
    for line in tokens:
        if line.value == 'if':
            if_num += 1
        elif line.value == 'case':
            if_num += 1
    return if_num

def getndi(tokens):
    num_declaration = 0
    for line in tokens:
        if line.value == 'int' or line.value == 'double' or line.value == 'float' or line.value == 'long' or line.value == 'short' or line.value == 'byte':
            num_declaration += 1
    return num_declaration

def add_nodes(G, dir_path):
    # num = 0
    for file in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file)
        if os.path.isfile(file_path):
            G.add_node(file)
            tokens = tokenize(file_path)
            if tokens:
                depth = getmdn(tokens)

                G.add_edge(file, 'mdn', weight = depth)
                if getmnp(tokens):
                    G.add_edge(file, 'mnp', weight =getmnp(tokens))
                if getloop(tokens):
                    G.add_edge(file, 'loop', weight = getloop(tokens))
                if getnfci(tokens):
                    G.add_edge(file, 'nfci', weight = getnfci(tokens))
                if getndi(tokens):
                    G.add_edge(file, 'ndi', weight =getndi(tokens))
            else:
                with open('./failed/metrics.txt', 'a+') as failed_file:
                    failed_file.write(file_path + '\n')
                    failed_file.close
            parsed_member = parsed_mem(file_path)
            if parsed_member:
                if getxmet(parsed_member):
                    G.add_edge(file, 'xmet', weight = getxmet(parsed_member))
                if getvref(parsed_member):
                    G.add_edge(file, 'vref', weight = getvref(parsed_member))
                if getvdec(parsed_member):
                    G.add_edge(file, 'vdec', weight = getvdec(parsed_member))
                if getnos(parsed_member):
                    G.add_edge(file, 'nos', weight = getnos(parsed_member))
                if getnopr(parsed_member):
                    G.add_edge(file, 'nopr', weight = getnopr(parsed_member))
                if getnoa(parsed_member):
                    G.add_edge(file, 'noa', weight = getnoa(parsed_member))
                if getnexp(parsed_member):
                    G.add_edge(file, 'nexp', weight = getnexp(parsed_member))
                if getnand(parsed_member):
                    G.add_edge(file, 'nand', weight = getnand(parsed_member))
                if getlmet(parsed_member):
                    G.add_edge(file, 'lmet', weight = getlmet(parsed_member))
                if getexct(parsed_member):
                    G.add_edge(file, 'exct', weight = getexct(parsed_member))
                if getexcr(parsed_member):
                    G.add_edge(file, 'excr', weight = getexcr(parsed_member))
                if getcref(parsed_member):
                    G.add_edge(file, 'cref', weight = getcref(parsed_member))
                if getcomp(parsed_member):
                    G.add_edge(file, 'comp', weight = getcomp(parsed_member))
                if getcast(parsed_member):
                    G.add_edge(file, 'cast', weight = getcast(parsed_member))
                if getnbltrl(parsed_member):
                    G.add_edge(file, 'nbltrl', weight = getnbltrl(parsed_member))
                if getncltrl(parsed_member):
                    G.add_edge(file, 'ncltrl', weight = getncltrl(parsed_member))
                if getnsltrl(parsed_member):
                    G.add_edge(file, 'nsltrl', weight = getnsltrl(parsed_member))
                if getnnltrl(parsed_member):
                    G.add_edge(file, 'nnltrl', weight = getnnltrl(parsed_member))
                if getnnulltrl(parsed_member):
                    G.add_edge(file, 'nnulltrl', weight = getnnulltrl(parsed_member))
                if gethvoc(parsed_member):
                    G.add_edge(file, 'hvoc', weight = gethvoc(parsed_member))
                if gethdif(parsed_member):
                    G.add_edge(file, 'hdif', weight = gethdif(parsed_member))
                if getheff(parsed_member):
                    G.add_edge(file, 'heff', weight = getheff(parsed_member))
            else:
                with open('./failed/metrics.txt', 'a+') as failed_file:
                    failed_file.write(file_path + '\n')
                    failed_file.close

    return G

# def test(G, dir_path, metrics_list):
#     for file in os.listdir(dir_path):
#         file_path = os.path.join(dir_path, file)
#         if os.path.isfile(file_path):
#             G.add_node(file)
#             for metrics in metrics_list:
#                 if metrics in [getmdn, getmnp, getloop, getnfci, getndi]:
#                     tokens = tokenize(file_path)
#                     if tokens:
#                         depth = metrics(tokens)
#                         # num += 1
#                         # print(str(num) + '   ' + file)
#                         G.add_edge(file, 'mdn', weight=depth)
#                         if metrics(tokens):
#                             G.add_edge(file, 'mnp', weight=getmnp(tokens))
#                         if metrics(tokens):
#                             G.add_edge(file, 'loop', weight=getloop(tokens))
#                         if metrics(tokens):
#                             G.add_edge(file, 'nfci', weight=getnfci(tokens))
#                         if metrics(tokens):
#                             G.add_edge(file, 'ndi', weight=getndi(tokens))
#                     else:
#                         with open('./failed/metrics.txt', 'a+') as failed_file:
#                             failed_file.write(file_path + '\n')
#                             failed_file.close
#
#     return G
