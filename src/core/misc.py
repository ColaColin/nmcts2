'''
Created on Apr 4, 2018

@author: cclausen
'''
import json

def openJson(path):
    with open(path) as f:
        parsed = json.load(f)
    return parsed

def writeJson(path, data, pretty = True):
    with open(path, "w") as f:
        f.truncate()
        if pretty:
            f.write(json.dumps(data, sort_keys=True, indent=4))
        else:
            f.write(json.dumps(data))