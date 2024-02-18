
## Parse Data Files Functions
    # by Brendan Wright

# summary
    # batch import, parse data files, store in array of dict, extract params from file name

## version history
    # [2018-12-27] 1v0 - initial build



## Import Libraries

# data array processing
import numpy as np

# filesystem parsing, regex handling
import os, re



## Import Raw Data
    # import raw data from file
# inputs
    # _file_path - full filepath [str]
    # _columns - ordered list of column headers [list]
    # _skip - number of header rows to skip [int]
# outputs
    # data_entry - imported data as labelled numpy arrays [dict]

def parse_raw_csv(_file_path, _columns, _skip = 0):

    # open raw data file
    with open(_file_path, 'rb') as file_:

        # load csv data from file into numpy array
        data = np.loadtxt(file_, delimiter = ',', skiprows = _skip, dtype = np.float32)

    # initialise data storage dict
    data_entry = {}

    # iterate data columns
    for i in range(len(_columns)):

        # store labelled column array in dict
        data_entry[ _columns[i] ] = data[:, i]

    # return data dict
    return data_entry



## Batch Parse Files
    # batched import of raw data from valid files in a directory
# inputs
    # _base_path - directory path containing files [str]
    # _columns - ordered list of column headers [list]
    # _file_type - file format as extension for parsing [str]
# outputs
    # data - array of data entry dicts containing data parsed from files

def batch_parse_files(_base_path, _columns, _file_type = 'csv'):

    # initialise dict for data storage
    data = []

    # get files list within directory
    files_list = [f for f in os.listdir(_base_path) if os.path.isfile(os.path.join(_base_path, f))]

    # iterate files
    for file_name in files_list:

        # build filename parse string with file type
        parse_string = '^(?P<file_name>.*)\.' + _file_type + '$'

        # parse filename from file
        file_name_parse = re.search(parse_string, file_name)

        # for files with valid file name
        if file_name_parse != None:

            # select file parse function by file type
            if _file_type == 'csv':

                # extract raw data from file (assume no headers)
                data_entry = parse_raw_csv( (_base_path + file_name), _columns )

            # initialise data entry dict, store parameters
            data_entry['file_name'] = file_name_parse.group('file_name')

            # store data entry in data array
            data.append(data_entry)

    # return parsed data array
    return data



## Parse Data Name
    # parse parameters from file name
# inputs
    # _data - reference to raw data array list [list]
    # _parse_string - name parse format regex string [str]
# outputs
    # none - data list updated with paramaters parsed from name, discard on parse error

def parse_data_name(_data, _parse_string):

    # initialise new data array
    data = []

    # extract list of parameters in parse string
    params = re.findall('<([^<>]*)>', _parse_string)

    # iterate data entries
    for i in range(len(_data)):

        data_entry = _data[i]

        # parse name
        re_name = re.search(_parse_string, data_entry['file_name'])

        # for files with valid file name
        if re_name is not None:

            # iterate each parameter to be extracted
            for param in params:

                # store parameter in data entry
                data_entry[param] = re_name.group(param)

            # store updated data entry in data array
            data.append(data_entry)

        # if name parse error
        else:
            # warn of parse error, entry discarded
            print( 'invalid name: ' + data_entry['file_name'] )

    # return new data array
    return data



## Cold Storage Functions
    # by Brendan Wright

# summary
    # store/load data to/from file using Pickle library


## version history
    # 1v0 - initial build
    # [2018-12-27] 1v1 - tidy and generalise



## Import Libraries

# dataset filesystem storage
import pickle



## Store Data
    # store data in binary file (pickle format)
# inputs
    # base path
    # data array
    # file name
# outputs
    # none, data stored in binary pickle file at base path / file name

def save_data_store(_data, _base_path, _file_name):
    # open binary file for writing
    with open(_base_path + _file_name, 'wb') as file:
        # dump pickle of data storage array to file
        pickle.dump(_data, file)



## Load Data
    # load data from binary file (pickle format)
# inputs
    # base path
    # file name
# outputs
    # data array loaded from binary pickle file at base path / file name

def load_data_store(_base_path, _file_name):
    # open binary file for reading
    with open(_base_path + _file_name, 'rb') as file:
        # load pickled data storage array
        data = pickle.load(file)
        return data
