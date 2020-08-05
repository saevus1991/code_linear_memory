import sys 
import socket


def add_path(name):
    """
    add certain paths that vary on different machines
    """
    # set up list of hosts with path
    host_list = {
        'nemesis.bcs.e-technik.tu-darmstadt.de': '/Users/christian/Documents/Code/',
        'nemesis.local': '/Users/christian/Documents/Code/',
        'gauss': '/home/cwildne/code/',
        'weyl': '/home/cwildne/code/',
        'kepler': '/home/cwildne/code/'
    }
    # construct path
    host_name = socket.gethostname()
    path = host_list[host_name] + name
    sys.path.append(path)
    return