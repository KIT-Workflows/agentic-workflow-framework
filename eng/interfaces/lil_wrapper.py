import json
import re
import functools
from typing import Union, Dict, List, Any
from IPython.display import display, Markdown


# Load the styles
with open('eng/assets/text_styling.json', 'r') as f:
    styles = json.load(f)

def apply_style(text, *style_names):
    style_codes = [styles['styles'].get(s) or styles['colors'].get(s) or styles['backgrounds'].get(s) for s in style_names]
    return f"{''.join(style_codes)}{text}{styles['styles']['reset']}"

def has_markdown(text):
    """
    Detect if the given text contains Markdown elements.
    
    Args:
    text (str): The input text to analyze.
    
    Returns:
    bool: True if Markdown elements are detected, False otherwise.
    """
    patterns = [
        r'\[.+?\]\(.+?\)',          # Links
        r'!\[.+?\]\(.+?\)',         # Images
        r'(?<!\\)(\*\*|__).+?\1',   # Bold
        r'(?<!\\)(\*|_).+?\1',      # Italic
        r'(?<!\\)(`).+?\1',         # Inline code
        r'^\s*#{1,6}\s',            # Headers
        r'^\s*[-*+]\s',             # Unordered lists
        r'^\s*\d+\.\s',             # Ordered lists
        r'^\s*>',                   # Blockquotes
        r'^\s*(`{3,}|~{3,})',       # Code blocks
        r'^-{3,}|^\*{3,}|^_{3,}',   # Horizontal rules
        r'\|.+\|.+\|',              # Tables
    ]
    combined_pattern = '|'.join(f'({p})' for p in patterns)
    if re.search(combined_pattern, text, re.MULTILINE):
        return True
    
    return False


def dict_to_string_r(input_dict: Union[Dict, List[Dict]], indent=0, indent_step=2) -> str:
    """
    Convert a dictionary to a formatted string representation.

    Args:
    input_dict (Union[Dict, List[Dict]]): The dictionary to convert.
    indent (int): The initial indentation level. Defaults to 0.
    indent_step (int): The number of spaces to indent at each level. Defaults to 2.

    Returns:
    str: The formatted string representation of the dictionary.
    """
    result = []
    for key, value in input_dict.items():
        # Create the indentation
        space = ' ' * (indent * indent_step)
        key = str(key)
        
        if isinstance(value, dict):
            # Recursively format nested dictionaries
            result.append(f"{space}{key}:\n{{")
            result.append(dict_to_string_r(value, indent + 1, indent_step))
            result.append(f"{space}}}")
        elif isinstance(value, list):
            # Format lists
            result.append(f"{space}{key}: [")
            for item in value:
                result.append(f"{space}  {item},")
            result.append(f"{space}]")
        elif isinstance(value, str):
            # Format strings
            result.append(f"{space}{key}:\n\"{value}\"")
        elif isinstance(value, (int, float)):
            # Format numbers
            result.append(f"{space}{key}:\n{value}")
        elif value is None:
            # Format None
            result.append(f"{space}{key}: None")
        else:
            # Format other types
            result.append(f"{space}{key}:\n{value}")
    
    return '\n'.join(result)


def process_output_to_str(result: Union[str, Dict, List[Dict], List[str], Any]) -> str:
    """
    Process different types of output and return a formatted string.
    
    Args:
    result: The output to process, which can be of various types.
    
    Returns:
    A formatted string representation of the output.
    """
    if isinstance(result, str):
        return result
    elif isinstance(result, dict):
        return dict_to_string_r(result)
    elif isinstance(result, list):
        if all(isinstance(item, dict) for item in result):
            return dict_to_string_r(result)
        elif all(isinstance(item, str) for item in result):
            return '\n'.join(result)
        else:
            return str(result)  # Fallback for mixed-type lists
    else:
        return str(result)

def stylish_print(markdown=False):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            print(apply_style(f"Calling `{func.__name__}`...", *styles['combinations']['highlight']))
            try:
                result = func(*args, **kwargs)
                if markdown:
                    if 'display' in globals() and 'Markdown' in globals():
                        display(Markdown(result))
                    else:
                        print("Markdown output not supported in this environment.")
                        print(result)
                else:
                    result_str = process_output_to_str(result)
                    print(apply_style(f"Output:\n{result_str}", *styles['combinations']['info']))
                return result
            except Exception as e:
                print(apply_style(f"Error in `{func.__name__}`: {str(e)}", *styles['combinations']['error']))
                raise
        return wrapper
    return decorator


def repeat_until_condition(condition = None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            while True:
                result = func(*args, **kwargs)
                if isinstance(result, bool):
                    if result:
                        return result
                elif condition(result):
                    return result
        return wrapper
    return decorator
