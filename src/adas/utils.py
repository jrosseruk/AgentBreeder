import sys

sys.path.append("src")


def extract_async_forward_as_string(file_path):
    """
    Extracts the async def forward method from the given Python file and returns it as a string.

    Args:
        file_path (str): Path to the Python file.

    Returns:
        str: The extracted async def forward method as a string.
    """
    with open(file_path, "r") as file:
        lines = file.readlines()

    forward_lines = []
    in_forward = False
    indent_level = None

    for line in lines:
        stripped_line = line.lstrip()

        # Check if the line starts the async def forward method
        if not in_forward and stripped_line.startswith("async def forward("):
            in_forward = True
            indent_level = len(line) - len(stripped_line)
            forward_lines.append(stripped_line.rstrip())
            continue

        if in_forward:
            current_indent = len(line) - len(stripped_line)

            # If we encounter a line with indentation less than or equal to the method definition, stop
            if stripped_line and current_indent <= indent_level:
                break

            # Remove the initial indentation
            if indent_level is not None:
                forward_lines.append(line[indent_level:].rstrip())
            else:
                forward_lines.append(line.rstrip())

    # Combine the lines into a single string
    forward_string = "\n".join(forward_lines)
    return forward_string


if __name__ == "__main__":

    file_path = "/adas/drop.py"
    forward_string = extract_async_forward_as_string(file_path)
    print(forward_string)
