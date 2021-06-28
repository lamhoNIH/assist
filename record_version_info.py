import os
from datetime import datetime

# File version.txt can be created during make using the main function
def main():

    stream = os.popen('git show -s --format="gitinfo: %h %ci"')
    output = stream.read().strip()
    if output.startswith('gitinfo:'):
        output = output.replace('gitinfo: ', '')
    else:
        output = "NA"

    print(output)
    
if __name__ == '__main__':
    main()