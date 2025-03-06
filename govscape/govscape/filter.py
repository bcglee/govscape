class Filter:
    def __init__(self, config : ServerConfig):
        self.embedding_directory = config.embedding_directory
        # this is a nested dict. 
        # key = filename, val = dictof the json (key = filter label, val = val of the filter label)
        self.file_json_dict = {}
        for root, _, files in os.walk(self.embedding_directory):
            for file in files:
                if file.endswith('.json'):
                    with open(file, "r") as file_json:
                        data = json.load(file_json)
                        directory = os.path.dirname(file)
                        self.file_json_dict[directory] = data

    # inputs: search_results = list of search results from server
    # filters = dict of filters to consider. key = filter, val = tuple that indicates range
    # returns: search_results refined after the filters
    def filter(search_results, filters):
        filtered_results = []
        for f in filters.keys():
            for sr in search_results:
                # get filter info from json with that specific filename.
                filename = sr["pdf"]
                f_val = self.file_json_dict[filename][f]
                    # check if within filter
                    if filters[f][0] == filters[f][1]:
                        if f_val == filters[f][0]
                            filtered_results.append(sr)
                    else:
                        if filters[f][0] <= f_val <= filters[f][1]:
                            filtered_results.append(sr)
        return filtered_results
       

       #TODO: just get the json on filter query 