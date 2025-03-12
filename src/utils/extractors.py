import re
import json


def extract_json_output(response: str):
    json_match = re.search(r"<output>(.*?)</output>", response, re.DOTALL)
    json_string = json_match.group(1).strip()
    parsed_json = json.loads(json_string)        
    return parsed_json

def extract_str_output(response: str):
    str_match = re.search(r"<output>(.*?)</output>", response, re.DOTALL)
    str_string = str_match.group(1).strip()
    return str_string