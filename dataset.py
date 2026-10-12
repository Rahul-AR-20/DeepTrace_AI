"""
Dataset builder for the Programming Error Analyzer.

There is no standard public dataset for this task, so this module generates a
synthetic one from realistic error-message templates (Python, Java, C/C++,
JavaScript/TypeScript, npm/pip, databases, networking, APIs).
Classes: Syntax, Runtime, Type, Dependency, Database, Network, API.

To improve the model, put REAL messages (Stack Overflow, your own logs, ...) in
data/real_errors.csv with columns: message,label  -- train.py merges them in.

Run:  python dataset.py        -> writes data/errors.csv
"""
import os
import random
import re

import pandas as pd

VARS = ["name", "user", "data", "items", "result", "count", "value", "text", "node", "config",
        "student", "total", "price", "index", "buffer", "response", "arr", "obj", "key", "temp"]
MODS = ["pandas", "numpy", "requests", "flask", "django", "scipy", "matplotlib", "torch",
        "tensorflow", "sklearn", "cv2", "bs4", "yaml", "lxml", "pytest", "express", "lodash",
        "react", "axios", "moment", "mysql", "psycopg2", "seaborn", "nltk"]
PKGS = ["org.apache.commons.lang3", "com.google.gson", "org.springframework.boot", "javax.servlet",
        "com.mysql.jdbc", "org.junit", "org.slf4j", "com.fasterxml.jackson"]
CLASSES = ["Student", "Employee", "Account", "Order", "User", "Node", "Parser", "Shape",
           "Product", "Invoice", "Config", "Logger"]
FUNCS = ["add", "compute", "process", "calculate", "getData", "parse", "save", "load", "update", "render"]
FILES = ["data.csv", "config.json", "input.txt", "output.log", "settings.yaml", "model.pkl",
         "users.db", "report.pdf"]
KEYWORDS = ["if", "for", "while", "def", "class", "else", "try"]
TYPES = ["int", "str", "float", "list", "dict", "bool", "NoneType"]
DBS = ["users_db", "prod_db", "postgres", "mysql", "mongodb", "redis", "app_db", "test_db"]
TABLES = ["users", "orders", "products", "accounts", "payments", "transactions", "profiles", "sessions"]
COLS = ["user_id", "email", "created_at", "status", "amount", "password_hash", "role", "balance"]
HOSTS = ["localhost", "127.0.0.1", "api.example.com", "db.internal.net", "auth.service.org", "cluster.mongodb.net"]
PORTS = ["5432", "3306", "27017", "6379", "8080", "443", "80", "3000"]
ENDPOINTS = ["/v1/users", "/api/v2/auth/login", "/v1/checkout", "/graphql", "/api/data", "/v1/tokens"]

TEMPLATES = {
    "Syntax": [
        "SyntaxError: invalid syntax",
        "SyntaxError: unexpected EOF while parsing",
        "IndentationError: expected an indented block after '<kw>' statement on line <n>",
        "IndentationError: unindent does not match any outer indentation level",
        "IndentationError: unexpected indent",
        "SyntaxError: unterminated string literal (detected at line <n>)",
        "SyntaxError: '(' was never closed",
        "SyntaxError: invalid syntax. Perhaps you forgot a comma?",
        "SyntaxError: EOL while scanning string literal",
        "SyntaxError: positional argument follows keyword argument",
        "SyntaxError: cannot assign to literal",
        "SyntaxError: expected ':'",
        "Main.java:<n>: error: ';' expected",
        "Main.java:<n>: error: ')' expected",
        "error: class, interface, or enum expected",
        "error: reached end of file while parsing",
        "error: illegal start of expression",
        "error: not a statement",
        "error: unclosed string literal",
        "error: expected ';' before '}' token",
        "error: expected ')' before ';' token",
        "error: expected declaration or statement at end of input",
        "Uncaught SyntaxError: Unexpected token '}'",
        "Uncaught SyntaxError: Unexpected identifier '<var>'",
        "SyntaxError: Unexpected end of input",
        "SyntaxError: Missing initializer in const declaration",
        "SyntaxError: Invalid or unexpected token",
        "SyntaxError: missing ) after argument list",
    ],
    "Runtime": [
        'java.lang.NullPointerException: Cannot invoke "String.length()" because "<var>" is null',
        'java.lang.NullPointerException: Cannot read field "<var>" because "<var>" is null',
        "java.lang.NullPointerException",
        "java.lang.ArrayIndexOutOfBoundsException: Index <n> out of bounds for length <m>",
        "java.lang.IndexOutOfBoundsException: Index: <n>, Size: <m>",
        "java.lang.ArithmeticException: / by zero",
        "java.lang.NumberFormatException: For input string: \"<var>\"",
        "java.lang.StackOverflowError",
        "java.lang.OutOfMemoryError: Java heap space",
        "java.util.ConcurrentModificationException",
        "java.util.NoSuchElementException: No value present",
        "java.io.FileNotFoundException: <file> (No such file or directory)",
        "ZeroDivisionError: division by zero",
        "IndexError: list index out of range",
        "KeyError: '<var>'",
        "RecursionError: maximum recursion depth exceeded",
        "MemoryError",
        "FileNotFoundError: [Errno 2] No such file or directory: '<file>'",
        "PermissionError: [Errno 13] Permission denied: '<file>'",
        "NameError: name '<var>' is not defined",
        "UnboundLocalError: local variable '<var>' referenced before assignment",
        "ValueError: invalid literal for int() with base 10: '<var>'",
        "AttributeError: 'NoneType' object has no attribute '<var>'",
        "StopIteration",
        "Segmentation fault (core dumped)",
        "Floating point exception (core dumped)",
        "terminate called after throwing an instance of 'std::out_of_range'",
        "terminate called after throwing an instance of 'std::bad_alloc'",
        "Uncaught ReferenceError: <var> is not defined",
        "RangeError: Maximum call stack size exceeded",
    ],
    "Type": [
        "TypeError: unsupported operand type(s) for +: '<type>' and '<type>'",
        "TypeError: can only concatenate str (not \"<type>\") to str",
        "TypeError: '<type>' object is not callable",
        "TypeError: '<type>' object is not iterable",
        "TypeError: '<type>' object is not subscriptable",
        "TypeError: object of type '<type>' has no len()",
        "TypeError: list indices must be integers or slices, not <type>",
        "TypeError: <func>() takes <n> positional arguments but <m> were given",
        "TypeError: <func>() missing 1 required positional argument: '<var>'",
        "TypeError: <var> is not a function",
        "TypeError: Assignment to constant variable.",
        "TypeError: Cannot convert undefined or null to object",
        "error: incompatible types: String cannot be converted to int",
        "error: incompatible types: possible lossy conversion from double to int",
        "error: incompatible types: int cannot be converted to boolean",
        "error: no suitable method found for <func>(int,String)",
        "error: bad operand types for binary operator '+'",
        "error: array required, but int found",
        "error: invalid conversion from 'int' to 'char*'",
        "error: cannot convert 'const char*' to 'int' in assignment",
        "error: no match for 'operator+' (operand types are 'std::string' and 'int')",
        "error TS2322: Type 'string' is not assignable to type 'number'.",
        "error TS2345: Argument of type 'string' is not assignable to parameter of type 'number'.",
        "java.lang.ClassCastException: class <cls> cannot be cast to class <cls>",
        "error: incompatible pointer types assigning to 'int *' from 'char *'",
    ],
    "Dependency": [
        "ModuleNotFoundError: No module named '<mod>'",
        "ImportError: No module named <mod>",
        "ImportError: cannot import name '<var>' from '<mod>'",
        "ImportError: DLL load failed while importing <mod>: The specified module could not be found.",
        "ImportError: libcuda.so.1: cannot open shared object file: No such file or directory",
        "Error: Cannot find module '<mod>'",
        "Module not found: Error: Can't resolve '<mod>' in '/src'",
        "java.lang.ClassNotFoundException: <pkg>.<cls>",
        "java.lang.NoClassDefFoundError: <pkg>/<cls>",
        "java.lang.UnsatisfiedLinkError: no <mod> in java.library.path",
        "error: package <pkg> does not exist",
        "fatal error: <mod>.h: No such file or directory",
        "undefined reference to `<func>'",
        "ld: library not found for -l<mod>",
        "npm ERR! code E404 npm ERR! 404 Not Found - GET https://registry.npmjs.org/<mod>",
        "npm ERR! peer dep missing: <mod>@^<n>.0.0",
        "ERROR: Could not find a version that satisfies the requirement <mod>",
        "ERROR: No matching distribution found for <mod>",
        "pkg_resources.DistributionNotFound: The '<mod>' distribution was not found",
        "Could not resolve dependency: peer <mod>@\"<n>.x\"",
        "Could not find artifact <pkg>:<mod>:jar:<n>.<m> in central",
        "npm ERR! code ERESOLVE unable to resolve dependency tree for <mod>",
        "Poetry: Could not find a matching version of package <mod>",
        "cargo error: no matching package named `<mod>` found",
        "go: module <mod>: git ls-remote failed: exit status 128",
    ],
    "Database": [
        "psycopg2.OperationalError: could not connect to server: Connection refused (0x0000274D/10061)",
        "psycopg2.OperationalError: FATAL: password authentication failed for user \"<var>\"",
        "psycopg2.errors.UndefinedTable: relation \"<table>\" does not exist",
        "psycopg2.errors.UndefinedColumn: column \"<col>\" of relation \"<table>\" does not exist",
        "sqlite3.OperationalError: no such table: <table>",
        "sqlite3.OperationalError: table <table> has no column named <col>",
        "sqlite3.OperationalError: database is locked",
        "sqlite3.IntegrityError: UNIQUE constraint failed: <table>.<col>",
        "pymongo.errors.ServerSelectionTimeoutError: No servers found yet, trying at <host>:<port>",
        "pymongo.errors.OperationFailure: Authentication failed for user '<var>'",
        "redis.exceptions.ConnectionError: Error 111 connecting to <host>:<port>. Connection refused.",
        "java.sql.SQLException: Access denied for user '<var>'@'<host>' (using password: YES)",
        "java.sql.SQLSyntaxErrorException: Table '<table>' doesn't exist",
        "java.sql.SQLIntegrityConstraintViolationException: Duplicate entry '<var>' for key 'PRIMARY'",
        "com.mysql.cj.jdbc.exceptions.CommunicationsException: Communications link failure",
        "org.hibernate.exception.SQLGrammarException: could not execute query for table <table>",
        "sqlalchemy.exc.OperationalError: (sqlite3.OperationalError) no such table: <table>",
        "sqlalchemy.exc.ProgrammingError: (psycopg2.errors.UndefinedColumn) column <table>.<col> does not exist",
        "sqlalchemy.exc.IntegrityError: (psycopg2.errors.UniqueViolation) duplicate key value violates unique constraint \"<col>\"",
        "MongoServerError: E11000 duplicate key error collection: <db>.<table> index: <col>_1 dup key",
        "MongooseError: Operation `<table>.find()` buffered timed out after 10000ms",
        "Django DB Error: django.db.utils.OperationalError: no such table: <table>",
        "Django DB Error: django.db.utils.IntegrityError: UNIQUE constraint failed: <table>.<col>",
        "MySQLdb.OperationalError: (2003, \"Can't connect to MySQL server on '<host>' (10061)\")",
    ],
    "Network": [
        "requests.exceptions.ConnectionError: HTTPSConnectionPool(host='<host>', port=<port>): Max retries exceeded with url: <endpoint>",
        "requests.exceptions.ConnectTimeout: HTTPConnectionPool(host='<host>', port=<port>): Read timed out.",
        "requests.exceptions.SSLError: HTTPSConnectionPool(host='<host>', port=443): Max retries exceeded (Caused by SSLError(SSLCertVerificationError(1, '[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: self-signed certificate')))",
        "socket.gaierror: [Errno 11001] getaddrinfo failed for host '<host>'",
        "TimeoutError: [Errno 110] Connection timed out to <host>:<port>",
        "ConnectionResetError: [Errno 104] Connection reset by peer from <host>",
        "ConnectionRefusedError: [Errno 111] Connection refused connecting to <host>:<port>",
        "net::ERR_CONNECTION_REFUSED at <host>",
        "net::ERR_NAME_NOT_RESOLVED for <host>",
        "net::ERR_CONNECTION_TIMED_OUT loading <host><endpoint>",
        "net::ERR_CERT_AUTHORITY_INVALID",
        "Access to fetch at 'https://<host><endpoint>' from origin 'http://localhost:3000' has been blocked by CORS policy: No 'Access-Control-Allow-Origin' header is present on the requested resource.",
        "Error: connect ECONNREFUSED <host>:<port>",
        "Error: getaddrinfo ENOTFOUND <host>",
        "Error: ETIMEDOUT connecting to <host>:<port>",
        "Error: read ECONNRESET from socket",
        "java.net.ConnectException: Connection refused (Connection refused)",
        "java.net.UnknownHostException: <host>",
        "java.net.SocketTimeoutException: Read timed out connecting to <host>",
    ],
    "API": [
        "requests.exceptions.HTTPError: 401 Client Error: Unauthorized for url: https://<host><endpoint>",
        "requests.exceptions.HTTPError: 403 Client Error: Forbidden for url: https://<host><endpoint>",
        "requests.exceptions.HTTPError: 404 Client Error: Not Found for url: https://<host><endpoint>",
        "requests.exceptions.HTTPError: 429 Client Error: Too Many Requests for url: https://<host><endpoint>",
        "requests.exceptions.HTTPError: 500 Server Error: Internal Server Error for url: https://<host><endpoint>",
        "requests.exceptions.HTTPError: 502 Server Error: Bad Gateway for url: https://<host><endpoint>",
        "requests.exceptions.HTTPError: 503 Server Error: Service Unavailable for url: https://<host><endpoint>",
        "json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)",
        "AxiosError: Request failed with status code 401 (Unauthorized)",
        "AxiosError: Request failed with status code 403 (Forbidden)",
        "AxiosError: Request failed with status code 404 (Not Found)",
        "AxiosError: Request failed with status code 429 (Too Many Requests)",
        "AxiosError: Request failed with status code 500 (Internal Server Error)",
        "AxiosError: Request failed with status code 502 (Bad Gateway)",
        "GraphQL error: Cannot query field \"<func>\" on type \"<cls>\"",
        "GraphQL error: Variable \"$<var>\" of required type \"<type>!\" was not provided.",
        "APIError: Invalid API Key provided in Authorization header",
        "APIError: Rate limit exceeded. Please wait <n> seconds before retrying.",
        "APIError: Missing required parameter '<var>' in API request body",
        "APIError: Malformed JWT token / Unauthorized access to <endpoint>",
    ],
}

# (prefix, suffix) wrappers shared by ALL classes so they carry no label information
WRAPPERS = [
    ("", ""),
    ("", ""),
    ('Traceback (most recent call last):\n  File "main.py", line <n>, in <module>\n', ""),
    ('Exception in thread "main" ', "\n\tat Main.main(Main.java:<n>)"),
    ("", "\n    at Object.<anonymous> (/app/index.js:<n>:<m>)"),
    ("[ERROR] ", ""),
    ("ERROR: ", ""),
    ("main.cpp:<n>:<m>: ", ""),
    ("Build failed: ", ""),
    ("", " (line <n>)"),
    ("", " in <file>"),
]


def _fill(template, rng):
    values = {
        "var": rng.choice(VARS), "mod": rng.choice(MODS), "pkg": rng.choice(PKGS),
        "cls": rng.choice(CLASSES), "func": rng.choice(FUNCS), "file": rng.choice(FILES),
        "kw": rng.choice(KEYWORDS), "type": rng.choice(TYPES),
        "db": rng.choice(DBS), "table": rng.choice(TABLES), "col": rng.choice(COLS),
        "host": rng.choice(HOSTS), "port": rng.choice(PORTS), "endpoint": rng.choice(ENDPOINTS),
        "n": str(rng.randint(1, 99)), "m": str(rng.randint(1, 99)),
    }
    return re.sub(r"<(var|mod|pkg|cls|func|file|kw|type|db|table|col|host|port|endpoint|n|m)>",
                  lambda m: values[m.group(1)], template)


def make_message(label, rng):
    msg = _fill(rng.choice(TEMPLATES[label]), rng)
    if rng.random() < 0.55:
        pre, suf = rng.choice(WRAPPERS)
        msg = _fill(pre, rng) + msg + _fill(suf, rng)
    if rng.random() < 0.10:
        msg = msg.lower()
    return msg


def build_dataset(per_class=1200, seed=42):
    rng = random.Random(seed)
    rows = []
    for label in TEMPLATES:
        seen, tries = set(), 0
        while len(seen) < per_class and tries < per_class * 30:
            seen.add(make_message(label, rng))
            tries += 1
        rows += [(m, label) for m in seen]
    df = pd.DataFrame(rows, columns=["message", "label"])
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def load_dataset(per_class=1200, seed=42, data_dir="data"):
    df = build_dataset(per_class, seed)
    real = os.path.join(data_dir, "real_errors.csv")
    if os.path.exists(real):
        extra = pd.read_csv(real)[["message", "label"]].dropna()
        print(f"Merging {len(extra)} real examples from {real}")
        df = pd.concat([df, extra], ignore_index=True)
    return df.drop_duplicates("message").sample(frac=1, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    d = load_dataset()
    d.to_csv("data/errors.csv", index=False)
    print(d.label.value_counts())
    print(d.head(8).to_string())

