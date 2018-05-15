
#
# ########################################################################################## #
#                                                                                            #
#   DATeS: Data Assimilation Testing Suite.                                                  #
#                                                                                            #
#   Copyright (C) 2016  A. Sandu, A. Attia, P. Tranquilli, S.R. Glandon,                     #
#   M. Narayanamurthi, A. Sarshar, Computational Science Laboratory (CSL), Virginia Tech.    #
#                                                                                            #
#   Website: http://csl.cs.vt.edu/                                                           #
#   Phone: 540-231-6186                                                                      #
#                                                                                            #
#   This program is subject to the terms of the Virginia Tech Non-Commercial/Commercial      #
#   License. Using the software constitutes an implicit agreement with the terms of the      #
#   license. You should have received a copy of the Virginia Tech Non-Commercial License     #
#   with this program; if not, please contact the computational Science Laboratory to        #
#   obtain it.                                                                               #
#                                                                                            #
# ########################################################################################## #
#   A downloader class that can be used to retrieve files from URL.                          #
#   e.g. sample files for QG model.                                                          #
# ########################################################################################## #
# 


"""
    A module providing classes and functions that handle url-related functionalities; such as downloading files, etc.
"""


from sys import stdout
from sys import exit as terminate
from time import time
import urllib
import urllib2


class URLDownload:
    """
    Download a file. A simple class to download files using urllib
    
    Args:
        link:
        file_name:
        checksum:
        download_immediately:
    
    """
    
    def __init__(self, link=None, file_name=None, checksum='md5', download_immediately=False):
        
        if checksum.lower() not in ['md5', 'sha1', 'sha224', 'sha256', 'sha384', 'sha512']:
            raise ValueError("Unrecognized checksum [%s]" % checksum)
        else:
            self.checksum = checksum.lower()

        self.file_link = link

        if file_name is not None:
            self.file_name = file_name
        elif link is not None:
            if '/' in self.file_link:
                self.file_name = self.file_link.split("/")[len(self.file_link.split('/')) - 1]
            else:
                self.file_name = link
        else:
            if download_immediately:
                download_immediately = False

        # retrieve the file from the given url upon initialization
        if download_immediately and link is not None:
            self.download(file_link=link, file_name=file_name)
        else:
            pass

    def download(self, file_link=None, file_name=None, print_summary=False, return_summary=False):
        """
        Download a file from web with showing progression and hash
        
        Args:
            file_link:
            file_name:
            print_summary:
            return_summary:
            
        """
        if file_link is None:
            raise ValueError("URL must be passed as a first argument!")
        else:
            if file_name is None:
                if '/' in file_link:
                    file_name = file_link.split("/")[len(file_link.split('/')) - 1]
                else:
                    file_name = file_link

        # Check for file first:
        try:
            f = urllib2.urlopen(file_link)
            f.close()
        except urllib2.HTTPError:
            terminate("\n>>> Failed to download the requested file:\n"
                      ">>> File Not Found. 404 Error returned!\n"
                      ">>> Please check the url: %s\n" % file_link)

        # File is available. Start downloading:
        timer = time()
        checksum = self.checksum

        urllib.urlretrieve(file_link, file_name, self.hook)
        if print_summary:
            print("\nFile name\t= %s\n ETA\t\t= %i second(s)\n%s checksum\t= %s\n" % (file_name,
                                                                                      int(time() - timer),
                                                                                      checksum
                                                                                      )
                  )
        else:
            pass

        if return_summary:
            out_dic = dict(file_name=file_name,
                           time_elapsed=int(timer - time()),
                           checksum=checksum
                           )
            return out_dic
        else:
            pass

    def hook(self, *data):
        """
        This hook function will be called once on establishment of the network connection and once after
        each block read thereafter.
        The hook will be passed three arguments; a count of blocks transferred so far, a block size in bytes,
        and the total size of the file. The third argument may be -1 on older FTP servers which do not return
        a file size in response to a retrieval request.
        """
        if data[-1] != -1:
            file_size = int(data[2] / 1024.0)
            if file_size >= 1024.0**2:
                file_size_unit = 'GB'
                file_size /= (1024.0**2)
            elif file_size > 1024.0:
                file_size_unit = 'MB'
                file_size /= (1024.0)
            else:
                file_size_unit = 'KB'
        else:
            file_size = None

        downloaded_so_far = data[0]*data[1] / 1024.0  # in KB.
        if downloaded_so_far >= 1024.0**2:
            downloaded_so_far_unit = 'GB'
            downloaded_so_far /= (1024.0**2)
        elif downloaded_so_far > 1024.0:
            downloaded_so_far_unit = 'MB'
            downloaded_so_far /= 1024.0
        else:
            downloaded_so_far_unit = 'KB'

        if file_size is not None:
            percent = float(data[0]) * data[1] / data[2] * 100.0
            out_msg = u"\r... Retrieving [{0:3.1f}%]: {1:3.2f} {2:s}/ {3:3.2f} {4:s}  \t".format(percent,
                                                                                                 downloaded_so_far,
                                                                                                 downloaded_so_far_unit,
                                                                                                 file_size,
                                                                                                 file_size_unit)
        else:
            out_msg = u"\r... Downloading file : {0:3.2f} {1:s} \t".format(downloaded_so_far,
                                                                           downloaded_so_far_unit)
        #
        stdout.write(out_msg)
        stdout.flush()
        #
        
