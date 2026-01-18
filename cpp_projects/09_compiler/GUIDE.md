# Compiler for a Simple Language - Implementation Guide

## What is This Project?

Build a complete compiler that translates a high-level language to x86-64 assembly. This project teaches compiler design, code generation, optimization, and gives you deep understanding of how programming languages work.

## Why Build This?

- Understand how compilers transform source code to machine code
- Learn parsing, semantic analysis, and code generation
- Master intermediate representations and optimizations
- Implement type systems and symbol tables
- Build the foundation of all programming languages

---

## Architecture Overview

```
┌──────────────────────────────────────────┐
│         Source Code (.lang)              │
└──────────────┬───────────────────────────┘
               │
┌──────────────▼───────────────────────────┐
│     Lexer (Tokenization)                 │
│  "if (x > 5)" → [IF, LPAREN, ID, GT, ...]│
└──────────────┬───────────────────────────┘
               │
┌──────────────▼───────────────────────────┐
│     Parser (AST Construction)            │
│  Tokens → Abstract Syntax Tree           │
└──────────────┬───────────────────────────┘
               │
┌──────────────▼───────────────────────────┐
│  Semantic Analysis (Type Checking)       │
│  Resolve symbols, check types            │
└──────────────┬───────────────────────────┘
               │
┌──────────────▼───────────────────────────┐
│  IR Generation (Three-Address Code)      │
│  High-level → Intermediate Representation│
└──────────────┬───────────────────────────┘
               │
┌──────────────▼───────────────────────────┐
│  Optimization Passes                     │
│  Dead code, constant folding, inlining   │
└──────────────┬───────────────────────────┘
               │
┌──────────────▼───────────────────────────┐
│  Code Generation (x86-64 Assembly)       │
│  IR → Assembly with register allocation  │
└──────────────┬───────────────────────────┘
               │
┌──────────────▼───────────────────────────┐
│     Assembler & Linker (External)        │
│  Assembly → Executable Binary            │
└──────────────────────────────────────────┘
```

---

## Implementation Hints

### 1. Lexer (Tokenization)

**What you need:**
Break source code into tokens (keywords, identifiers, operators, literals).

**Hint:**
```cpp
enum class TokenType {
    // Keywords
    IF, ELSE, WHILE, FOR, RETURN, INT, BOOL, VOID,

    // Operators
    PLUS, MINUS, STAR, SLASH, PERCENT,
    EQ, NE, LT, LE, GT, GE,
    ASSIGN, LOGICAL_AND, LOGICAL_OR, NOT,

    // Delimiters
    LPAREN, RPAREN, LBRACE, RBRACE, SEMICOLON, COMMA,

    // Literals & Identifiers
    INTEGER, IDENTIFIER,

    // Special
    END_OF_FILE, ERROR
};

struct Token {
    TokenType type;
    std::string lexeme;
    int line, column;
    int int_value; // For INTEGER tokens
};

class Lexer {
private:
    std::string source;
    size_t current = 0;
    int line = 1, column = 1;

public:
    Lexer(const std::string& src) : source(src) {}

    Token nextToken() {
        skipWhitespace();

        if (isAtEnd()) {
            return makeToken(TokenType::END_OF_FILE);
        }

        char c = advance();

        // Identifiers and keywords
        if (isalpha(c) || c == '_') {
            return identifier();
        }

        // Numbers
        if (isdigit(c)) {
            return number();
        }

        // Operators and delimiters
        switch (c) {
            case '+': return makeToken(TokenType::PLUS);
            case '-': return makeToken(TokenType::MINUS);
            case '*': return makeToken(TokenType::STAR);
            case '/': return makeToken(TokenType::SLASH);
            case '(': return makeToken(TokenType::LPAREN);
            case ')': return makeToken(TokenType::RPAREN);
            case '{': return makeToken(TokenType::LBRACE);
            case '}': return makeToken(TokenType::RBRACE);
            case ';': return makeToken(TokenType::SEMICOLON);
            case ',': return makeToken(TokenType::COMMA);

            case '=':
                if (match('=')) return makeToken(TokenType::EQ);
                return makeToken(TokenType::ASSIGN);

            case '!':
                if (match('=')) return makeToken(TokenType::NE);
                return makeToken(TokenType::NOT);

            case '<':
                if (match('=')) return makeToken(TokenType::LE);
                return makeToken(TokenType::LT);

            case '>':
                if (match('=')) return makeToken(TokenType::GE);
                return makeToken(TokenType::GT);
        }

        return makeToken(TokenType::ERROR);
    }

private:
    Token identifier() {
        int start = current - 1;
        while (isalnum(peek()) || peek() == '_') {
            advance();
        }

        std::string lexeme = source.substr(start, current - start);
        TokenType type = checkKeyword(lexeme);

        Token token = makeToken(type);
        token.lexeme = lexeme;
        return token;
    }

    Token number() {
        int start = current - 1;
        while (isdigit(peek())) {
            advance();
        }

        std::string lexeme = source.substr(start, current - start);
        Token token = makeToken(TokenType::INTEGER);
        token.lexeme = lexeme;
        token.int_value = std::stoi(lexeme);
        return token;
    }

    TokenType checkKeyword(const std::string& lexeme) {
        static std::map<std::string, TokenType> keywords = {
            {"if", TokenType::IF},
            {"else", TokenType::ELSE},
            {"while", TokenType::WHILE},
            {"for", TokenType::FOR},
            {"return", TokenType::RETURN},
            {"int", TokenType::INT},
            {"bool", TokenType::BOOL},
            {"void", TokenType::VOID}
        };

        auto it = keywords.find(lexeme);
        return (it != keywords.end()) ? it->second : TokenType::IDENTIFIER;
    }

    char advance() {
        column++;
        return source[current++];
    }

    char peek() {
        return isAtEnd() ? '\0' : source[current];
    }

    bool match(char expected) {
        if (peek() == expected) {
            advance();
            return true;
        }
        return false;
    }

    void skipWhitespace() {
        while (!isAtEnd()) {
            char c = peek();
            if (c == ' ' || c == '\t' || c == '\r') {
                advance();
            } else if (c == '\n') {
                line++;
                column = 0;
                advance();
            } else {
                break;
            }
        }
    }

    bool isAtEnd() {
        return current >= source.size();
    }

    Token makeToken(TokenType type) {
        Token token;
        token.type = type;
        token.line = line;
        token.column = column;
        return token;
    }
};
```

**Tips:**
- Handle single-line (`//`) and multi-line (`/* */`) comments
- Track line numbers for error messages
- Support string literals with escape sequences
- Add floating-point number support

### 2. Parser (AST Construction)

**What you need:**
Parse tokens into an Abstract Syntax Tree using recursive descent.

**Hint:**
```cpp
// AST Node types
struct ASTNode {
    virtual ~ASTNode() = default;
};

struct ExprNode : ASTNode {};

struct BinaryExprNode : ExprNode {
    std::unique_ptr<ExprNode> left;
    TokenType op;
    std::unique_ptr<ExprNode> right;
};

struct IntegerNode : ExprNode {
    int value;
};

struct VariableNode : ExprNode {
    std::string name;
};

struct StmtNode : ASTNode {};

struct IfStmtNode : StmtNode {
    std::unique_ptr<ExprNode> condition;
    std::unique_ptr<StmtNode> then_branch;
    std::unique_ptr<StmtNode> else_branch;
};

struct WhileStmtNode : StmtNode {
    std::unique_ptr<ExprNode> condition;
    std::unique_ptr<StmtNode> body;
};

struct ReturnStmtNode : StmtNode {
    std::unique_ptr<ExprNode> value;
};

struct FunctionNode : ASTNode {
    std::string name;
    std::vector<std::pair<std::string, std::string>> params; // (type, name)
    std::string return_type;
    std::vector<std::unique_ptr<StmtNode>> body;
};

class Parser {
private:
    Lexer& lexer;
    Token current_token;

public:
    Parser(Lexer& lex) : lexer(lex) {
        advance();
    }

    std::unique_ptr<FunctionNode> parseFunction() {
        // function_decl = type identifier '(' params ')' '{' statements '}'

        auto func = std::make_unique<FunctionNode>();

        func->return_type = expect(TokenType::INT).lexeme; // Simplified
        func->name = expect(TokenType::IDENTIFIER).lexeme;

        expect(TokenType::LPAREN);
        // Parse parameters...
        expect(TokenType::RPAREN);

        expect(TokenType::LBRACE);
        while (current_token.type != TokenType::RBRACE) {
            func->body.push_back(parseStatement());
        }
        expect(TokenType::RBRACE);

        return func;
    }

    std::unique_ptr<StmtNode> parseStatement() {
        if (current_token.type == TokenType::IF) {
            return parseIfStatement();
        } else if (current_token.type == TokenType::WHILE) {
            return parseWhileStatement();
        } else if (current_token.type == TokenType::RETURN) {
            return parseReturnStatement();
        } else {
            // Expression statement
            auto expr = parseExpression();
            expect(TokenType::SEMICOLON);
            // Wrap in statement node...
        }
    }

    std::unique_ptr<ExprNode> parseExpression() {
        return parseComparison();
    }

    std::unique_ptr<ExprNode> parseComparison() {
        auto expr = parseTerm();

        while (current_token.type == TokenType::LT ||
               current_token.type == TokenType::GT ||
               current_token.type == TokenType::EQ) {
            TokenType op = current_token.type;
            advance();
            auto right = parseTerm();

            auto binary = std::make_unique<BinaryExprNode>();
            binary->left = std::move(expr);
            binary->op = op;
            binary->right = std::move(right);

            expr = std::move(binary);
        }

        return expr;
    }

    std::unique_ptr<ExprNode> parseTerm() {
        auto expr = parseFactor();

        while (current_token.type == TokenType::PLUS ||
               current_token.type == TokenType::MINUS) {
            TokenType op = current_token.type;
            advance();
            auto right = parseFactor();

            auto binary = std::make_unique<BinaryExprNode>();
            binary->left = std::move(expr);
            binary->op = op;
            binary->right = std::move(right);

            expr = std::move(binary);
        }

        return expr;
    }

    std::unique_ptr<ExprNode> parseFactor() {
        auto expr = parsePrimary();

        while (current_token.type == TokenType::STAR ||
               current_token.type == TokenType::SLASH) {
            TokenType op = current_token.type;
            advance();
            auto right = parsePrimary();

            auto binary = std::make_unique<BinaryExprNode>();
            binary->left = std::move(expr);
            binary->op = op;
            binary->right = std::move(right);

            expr = std::move(binary);
        }

        return expr;
    }

    std::unique_ptr<ExprNode> parsePrimary() {
        if (current_token.type == TokenType::INTEGER) {
            auto node = std::make_unique<IntegerNode>();
            node->value = current_token.int_value;
            advance();
            return node;
        }

        if (current_token.type == TokenType::IDENTIFIER) {
            auto node = std::make_unique<VariableNode>();
            node->name = current_token.lexeme;
            advance();
            return node;
        }

        if (current_token.type == TokenType::LPAREN) {
            advance();
            auto expr = parseExpression();
            expect(TokenType::RPAREN);
            return expr;
        }

        throw std::runtime_error("Unexpected token");
    }

private:
    void advance() {
        current_token = lexer.nextToken();
    }

    Token expect(TokenType type) {
        if (current_token.type != type) {
            throw std::runtime_error("Unexpected token");
        }
        Token token = current_token;
        advance();
        return token;
    }
};
```

**Tips:**
- Use operator precedence for correct parsing
- Implement error recovery to continue after errors
- Support function calls and array indexing
- Add support for structs/classes

### 3. Semantic Analysis (Type Checking)

**What you need:**
Verify types are used correctly and resolve variable names.

**Hint:**
```cpp
enum class Type {
    INT, BOOL, VOID, ERROR
};

struct Symbol {
    std::string name;
    Type type;
    int offset; // Stack offset for locals
};

class SemanticAnalyzer {
private:
    std::map<std::string, Symbol> symbol_table;
    int current_offset = 0;

public:
    Type checkExpression(ExprNode* node) {
        if (auto* binary = dynamic_cast<BinaryExprNode*>(node)) {
            Type left_type = checkExpression(binary->left.get());
            Type right_type = checkExpression(binary->right.get());

            // Type checking rules
            if (binary->op == TokenType::PLUS ||
                binary->op == TokenType::MINUS ||
                binary->op == TokenType::STAR ||
                binary->op == TokenType::SLASH) {
                if (left_type != Type::INT || right_type != Type::INT) {
                    error("Arithmetic operators require integer operands");
                    return Type::ERROR;
                }
                return Type::INT;
            }

            if (binary->op == TokenType::LT ||
                binary->op == TokenType::GT ||
                binary->op == TokenType::EQ) {
                if (left_type != right_type) {
                    error("Comparison requires same types");
                    return Type::ERROR;
                }
                return Type::BOOL;
            }
        }

        if (auto* var = dynamic_cast<VariableNode*>(node)) {
            auto it = symbol_table.find(var->name);
            if (it == symbol_table.end()) {
                error("Undefined variable: " + var->name);
                return Type::ERROR;
            }
            return it->second.type;
        }

        if (auto* integer = dynamic_cast<IntegerNode*>(node)) {
            return Type::INT;
        }

        return Type::ERROR;
    }

    void declareVariable(const std::string& name, Type type) {
        if (symbol_table.count(name)) {
            error("Variable already declared: " + name);
            return;
        }

        Symbol symbol;
        symbol.name = name;
        symbol.type = type;
        symbol.offset = current_offset;
        current_offset += 8; // 8 bytes per variable

        symbol_table[name] = symbol;
    }

private:
    void error(const std::string& msg) {
        std::cerr << "Type error: " << msg << std::endl;
    }
};
```

**Tips:**
- Implement scope management (nested scopes)
- Add type inference for `auto` keyword
- Support function overloading
- Implement implicit type conversions

### 4. Intermediate Representation (Three-Address Code)

**What you need:**
Convert AST to simpler IR suitable for optimization.

**Hint:**
```cpp
struct TAC {
    enum class Op {
        ADD, SUB, MUL, DIV,
        LT, GT, EQ,
        ASSIGN, LABEL, GOTO, IF_FALSE,
        CALL, RETURN, PARAM
    };

    Op op;
    std::string arg1, arg2, result;
};

class IRGenerator {
private:
    std::vector<TAC> instructions;
    int temp_counter = 0;
    int label_counter = 0;

public:
    std::string generateIR(ExprNode* node) {
        if (auto* binary = dynamic_cast<BinaryExprNode*>(node)) {
            std::string left = generateIR(binary->left.get());
            std::string right = generateIR(binary->right.get());

            std::string result = newTemp();

            TAC::Op op;
            switch (binary->op) {
                case TokenType::PLUS: op = TAC::Op::ADD; break;
                case TokenType::MINUS: op = TAC::Op::SUB; break;
                case TokenType::STAR: op = TAC::Op::MUL; break;
                case TokenType::SLASH: op = TAC::Op::DIV; break;
            }

            instructions.push_back({op, left, right, result});
            return result;
        }

        if (auto* integer = dynamic_cast<IntegerNode*>(node)) {
            return std::to_string(integer->value);
        }

        if (auto* var = dynamic_cast<VariableNode*>(node)) {
            return var->name;
        }

        return "";
    }

    void generateIfStatement(IfStmtNode* node) {
        std::string cond = generateIR(node->condition.get());

        std::string else_label = newLabel();
        std::string end_label = newLabel();

        // if (!cond) goto else_label
        instructions.push_back({TAC::Op::IF_FALSE, cond, "", else_label});

        // then branch
        generateStatement(node->then_branch.get());
        instructions.push_back({TAC::Op::GOTO, "", "", end_label});

        // else branch
        instructions.push_back({TAC::Op::LABEL, "", "", else_label});
        if (node->else_branch) {
            generateStatement(node->else_branch.get());
        }

        instructions.push_back({TAC::Op::LABEL, "", "", end_label});
    }

private:
    std::string newTemp() {
        return "t" + std::to_string(temp_counter++);
    }

    std::string newLabel() {
        return "L" + std::to_string(label_counter++);
    }
};
```

**Tips:**
- Use SSA (Static Single Assignment) form
- Implement basic blocks and control flow graph
- Add support for phi functions
- Track liveness information

### 5. x86-64 Code Generation

**What you need:**
Translate IR to assembly with register allocation.

**Hint:**
```cpp
class CodeGenerator {
private:
    std::ostringstream output;
    std::map<std::string, std::string> register_map; // temp -> register

    const std::vector<std::string> registers = {
        "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9"
    };
    int next_register = 0;

public:
    std::string generate(const std::vector<TAC>& instructions) {
        // Function prologue
        output << ".globl main\n";
        output << "main:\n";
        output << "    push rbp\n";
        output << "    mov rbp, rsp\n";

        for (const auto& instr : instructions) {
            generateInstruction(instr);
        }

        // Function epilogue
        output << "    mov rsp, rbp\n";
        output << "    pop rbp\n";
        output << "    ret\n";

        return output.str();
    }

private:
    void generateInstruction(const TAC& instr) {
        switch (instr.op) {
            case TAC::Op::ADD: {
                std::string reg1 = allocateRegister(instr.arg1);
                std::string reg2 = allocateRegister(instr.arg2);
                std::string result_reg = allocateRegister(instr.result);

                output << "    mov " << result_reg << ", " << reg1 << "\n";
                output << "    add " << result_reg << ", " << reg2 << "\n";
                break;
            }

            case TAC::Op::SUB: {
                std::string reg1 = allocateRegister(instr.arg1);
                std::string reg2 = allocateRegister(instr.arg2);
                std::string result_reg = allocateRegister(instr.result);

                output << "    mov " << result_reg << ", " << reg1 << "\n";
                output << "    sub " << result_reg << ", " << reg2 << "\n";
                break;
            }

            case TAC::Op::RETURN: {
                std::string reg = allocateRegister(instr.arg1);
                output << "    mov rax, " << reg << "\n";
                break;
            }

            case TAC::Op::LABEL: {
                output << instr.result << ":\n";
                break;
            }

            case TAC::Op::GOTO: {
                output << "    jmp " << instr.result << "\n";
                break;
            }

            case TAC::Op::IF_FALSE: {
                std::string cond = allocateRegister(instr.arg1);
                output << "    cmp " << cond << ", 0\n";
                output << "    je " << instr.result << "\n";
                break;
            }
        }
    }

    std::string allocateRegister(const std::string& var) {
        // Simple register allocation (no spilling)
        if (register_map.count(var)) {
            return register_map[var];
        }

        // Check if it's a constant
        if (isdigit(var[0])) {
            return var; // Immediate value
        }

        // Allocate new register
        std::string reg = registers[next_register % registers.size()];
        next_register++;

        register_map[var] = reg;
        return reg;
    }
};
```

**Tips:**
- Implement graph coloring for register allocation
- Handle register spilling to stack
- Use calling conventions (System V AMD64 ABI)
- Add peephole optimizations

---

## Project Structure

```
09_compiler/
├── CMakeLists.txt
├── src/
│   ├── main.cpp
│   ├── lexer/
│   │   └── lexer.cpp
│   ├── parser/
│   │   ├── parser.cpp
│   │   └── ast.cpp
│   ├── semantic/
│   │   ├── type_checker.cpp
│   │   └── symbol_table.cpp
│   ├── ir/
│   │   ├── ir_generator.cpp
│   │   └── optimizer.cpp
│   └── codegen/
│       ├── x86_codegen.cpp
│       └── register_allocator.cpp
├── tests/
│   ├── test_lexer.cpp
│   ├── test_parser.cpp
│   └── test_codegen.cpp
└── examples/
    ├── hello.lang
    ├── fibonacci.lang
    └── quicksort.lang
```

---

## Resources

- Book: "Crafting Interpreters" by Robert Nystrom
- Book: "Modern Compiler Implementation in C" by Andrew Appel
- [LLVM Tutorial](https://llvm.org/docs/tutorial/)
- [x86-64 Assembly Guide](https://cs.brown.edu/courses/cs033/docs/guides/x64_cheatsheet.pdf)

Good luck building your compiler!
