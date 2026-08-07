from Tool import Tool


def calculator(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b

calculator_tool = Tool(
    "calculator",  # name
    "Multiply two integers.",  # description
    calculator,  # function to call
    [("a", "int"), ("b", "int")],  # inputs (names and types)
    "int",  # output
)

print(calculator_tool.to_string())
