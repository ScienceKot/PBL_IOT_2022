from configparser import ConfigParser

def config_fun(filename='config.ini'):
    # create a parser
    parser = ConfigParser()
    # read config file
    parser.read(filename)

    all_configs = {}
    # get section, default to postgresql
    for section in parser.sections():
        section_dict = {}
        params = parser.items(section)
        for param in params:
            section_dict[param[0]] = param[1]
        all_configs[section] = section_dict

    return all_configs