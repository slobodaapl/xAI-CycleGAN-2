# create a data class that ingests a file with the following structure:
# <name>=<value> where <value> can either be a "string" or a number
# the class creates properties for each name and assigns the value to it

import re


class Settings:
    """
    This class loads a settings file and creates properties for each name in the file.
    The values are either strings or numbers.
    If the name starts with a star, the value is saved to self.cfg_dict to be used for wandb config.
    """

    def __init__(self, path):
        self.path = path
        self.cfg_dict = {}
        self.load_settings()

    def load_settings(self):
        with open(self.path, 'r') as file:
            for line in file:
                # if line is empty or starts with a comment, skip it
                if not line.strip() or line.strip().startswith('#'):
                    continue

                if '=' in line:
                    name, value = line.split('=')
                    name = name.strip()
                    value = value.strip()

                    if re.match(r'^-?\d+$', value):
                        value = int(value)
                    elif re.match(r'^-?\d+\.\d+$', value):
                        value = float(value)
                    elif value == 'None':
                        value = None
                    elif value == 'True':
                        value = True
                    elif value == 'False':
                        value = False
                    else:
                        value = value.strip('"').strip("'")

                    # check if name has a star on the beginning and if so, save it to self.cfg_dict
                    if name.startswith('*'):
                        name = name[1:]
                        self.cfg_dict[name] = value

                    setattr(self, name, value)
