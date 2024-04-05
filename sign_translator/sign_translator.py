import csv

SIGNS_FILE = "sign_translator/signs.csv"

class SignTranslator:
    def __init__(self):
        self.signs = self.parseCSVFile(SIGNS_FILE)

    def parseCSVFile(self, filename, parse_numbers = False):
        '''
        Parses a CSV file located in te config directory and returns its contents as a dictionary.

        @param filename: The name of the csv file to parse.

        @return the contents of the parsed file.
        '''
        result = {}
        with open(filename) as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            for row in reader:
                result[row[0]] = row[1] if not parse_numbers else float(row[1])
        return result
    
    def get_sign(self, s):
        '''
        Returns the sign name for a given sign id.

        @param s: the sign id

        @return the sign name
        '''
        return self.signs[str(s)]