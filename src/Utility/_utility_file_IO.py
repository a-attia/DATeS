
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
#


"""
    A module providing utility functions required to handle files IO operations.
"""


import shutil
import os
import sys
import zipfile


# #
# def get_list_of_subdirectories(root_dir):
#     """
#     Retrieve a list of sub-directories .
#
#     Args:
#         root_dir: directory to start constructing sub-directories of.
#
#     Returns:
#         subdirs_list: a list containing subdirectories under the given root_dir.
#
#     Returns:
#         subdirs_list: list of subdirectories;
#             returns None if root_dir has no subdirectories.
#    """
#     #
#     if not os.path.isdir(root_dir):
#         raise IOError(" ['%s'] is not a valid directory!" % root_dir)
#
#     subdirs_list = []
#     for root, _, _ in os.walk(root_dir):
#         # '/.' insures that the iterator ignores any subdirectory of special directory such as '.git' subdirs.
#         # '__' insures that the iterator ignores any cashed subdirectory.
#         if not ('/.' in root or '__' in root):
#             # in case this is not the initial run. We don't want  to add duplicates to the system paths' list.
#             subdirs_list.append(root)
#
#         if len(subdirs_list) == 0:
#             subdirs_list = None
#
#     #
#     return subdirs_list
#

def get_list_of_subdirectories(root_dir, ignore_root=False, return_abs=False, ignore_special=True, recursive=True):
    """
    Retrieve a list of sub-directories .

    Args:
        root_dir: directory to start constructing sub-directories of.
        ignore_root: True/False; if True, the passed root_dir is ignored in the returned list
        return_ab: True/False; if True, returned paths are absolute, otherwise relative (to root_dir)
        ignore_special: Ture/False; if True, this function ignores special files (starting with . or __ ).

    Returns:
        subdirs_list: a list containing subdirectories under the given root_dir.

    Returns:
        subdirs_list: list of subdirectories; the subdires are absolute paths or relative paths based the passed root
            returns empty list if root_dir has no subdirectories.

    Raises:
        IOError: If the passed path/directory does not exist
    """
    #
    if not os.path.isdir(root_dir):
        raise IOError(" ['%s'] is not a valid directory!" % root_dir)

    subdirs_list = []
    _passed_path = os.path.abspath(root_dir)
    if os.path.isabs(root_dir):
        pass  # OK, it's an absolute path, good to go
    else:
        # the passed path is relative; convert back, and guarantee it's of the right format for later comparison
        _passed_path = os.path.relpath(_passed_path)

    if recursive:
        for root, _, _ in os.walk(_passed_path):
            # '/.' insures that the iterator ignores any subdirectory of special directory such as '.git' subdirs.
            # '__' insures that the iterator ignores any cashed subdirectory.
            if ignore_special and ('/.' in root or '__' in root):
                continue

            # in case this is not the initial run. We don't want  to add duplicates to the system paths' list.
            if ignore_root and _passed_path == root:
                pass
            else:
                if return_abs:
                    subdirs_list.append(os.path.abspath(root))
                else:
                    subdirs_list.append(os.path.relpath(root))
    else:
        _sub = next(os.walk(_passed_path))[1]
        if not ignore_root:
            if return_abs:
                subdirs_list.append(os.path.abspath(_passed_path))
            else:
                subdirs_list.append(os.path.relpath(_passed_path))
        for d in _sub:
            if return_abs:
                subdirs_list.append(os.path.join(os.path.abspath(_passed_path), d))
            else:
                subdirs_list.append(os.path.join(os.path.relpath(_passed_path), d))

    #
    return subdirs_list

#
def get_list_of_files(root_dir, recursive=False, return_abs=True, extension=None):
    """
    Retrieve a list of files in the passed root_dir, (and optionally in all sub-directories).
    This function ignores special files (starting with . or __ ).

    Args:
        root_dir: directory to start constructing sub-directories of.
        recursive: True/False; if True, the passed files are found in subdirectories as well
        return_ab: True/False; if True, returned paths are absolute, otherwise relative (to root_dir)

    Returns:
        files_list: a list containing absolute (or relative) under the given root_dir.

    Raises:
        IOError: If the passed path/directory does not exist
   """
    #
    if not os.path.isdir(root_dir):
        raise IOError(" ['%s'] is not a valid directory!" % root_dir)
    else:
        _passed_abs_path = os.path.abspath(root_dir)
        _passed_rel_path = os.path.relpath(root_dir)

    if extension is None:
        _ext = extension
    else:
        _ext = extension.lower().lstrip('* .').rstrip(' ')
    #
    files_list = []
    if recursive:
        dirs_list = get_list_of_subdirectories(root_dir=_passed_abs_path, return_abs=False, ignore_special=False)
    else:
        dirs_list = [_passed_rel_path]

    for dir_name in dirs_list:
        loc_files = os.listdir(dir_name)
        #
        for fle in loc_files:
            file_rel_path = os.path.relpath(os.path.join(dir_name, fle))  # relative path of file
            if os.path.isfile(file_rel_path):  # To ignore directories...
                if not ('/.' in fle or '__' in fle):
                    # Add this file to the aggregated list:
                    if _ext is not None:
                        head, tail = os.path.splitext(file_rel_path)
                        if not(re.match(r'\A(.)*%s\Z' % _ext, tail, re.IGNORECASE)):
                            continue
                    if return_abs:
                        files_list.append(os.path.abspath(file_rel_path))
                    else:
                        files_list.append(file_rel_path)
            else:  # not a file; pass
                pass  # or continue

    return files_list


#
def try_file_name(directory, file_prefix, extension=None):
    """
    Try to find a suitable file name file_prefix_<number>.<extension>

    Args:
        directory:
        file_prefix:
        extension:

    Returns:
        file_name:

    """
    #
    if not os.path.isdir(directory):
        raise IOError(" ['%s'] is not a valid directory!" % directory)

    if not directory.endswith('/'):
        directory += '/'

    if extension is None:
        file_name = file_prefix
    else:
        file_name = file_prefix + '.'+extension.strip('. ')

    if not os.path.isfile(os.path.join(directory, file_name)):
        pass
    else:
        #
        success = False
        counter = 0
        while not success:
            if extension is None:
                file_name = file_prefix +'_' + str(counter)
            else:
                file_name = file_prefix +'_' + str(counter) + '.' + extension.strip('. ')

            if not (os.path.isfile(directory + file_name) ):
                success = True
                break
            else:
                pass
            counter += 1
    #
    return file_name

#
def cleanup_directory(directory_name, parent_path, backup_existing=True):
    """
    Try to find the directory name under the parent path. I.e. parent_path/directory
    IF the directory does not exist, create it, otherwise either delete it's contents or back them up.
    If backup_existing is True and zip_backup is true the backup is archived as *.zip file.

    Args:
        directory_name:
        parent_path:
        backup_existing:

    """
    #
    directory_full_path = os.path.join(parent_path, directory_name.strip('/ \\'))
    #
    if not os.path.isdir(parent_path) or not os.path.isdir(directory_full_path):
        os.makedirs(directory_full_path)
    else:
        if backup_existing:
            zip_dir(path=directory_full_path)
        else:
            pass
        # Remove the leaf directory only, keeping the parent path untouched.
        # the directory is recreated instead of searching for subdirectories and files.
        shutil.rmtree(directory_full_path)
        os.makedirs(directory_full_path)
        #


def zip_dir(path, output_location=None, save_full_path=False, allowZip64=True):
    """
    Backup a directory in a zip archive in the given 'output_location'.
    If an archive with the same name in the output_location exists, a proper number-suffix will replace the archive name.

    Args:
        path: the path of the directory to zip. All files and subdirectory in the leaf directory will be archived.
        output_location: where to save the zip archive
        save_full_path: if true the whole path will be traced while archiving.
        allowZip64: allow zipping large files (> 2GB)

    """
    assert isinstance(path, str)
    path = path.rstrip('/ ')
    if not os.path.isdir:
        raise IOError("The directory you want to archive does not exist or not a valid directory!")
    else:
        # check if there is an archive with the same name in the given directory
        # get proper archive name:
        archive_extension = 'zip'
        if output_location is not None:
            if not os.path.isdir(output_location):
                raise IOError("the passed output_location' is not a valid directory!")
            else:
                archive_parent_dir = output_location
                parent_dir, target_dir = os.path.split(path)
                file_name = target_dir
        else:
            parent_dir, target_dir = os.path.split(path)
            if len(parent_dir) == 0:
                raise IOError("Parent directory is empty. Do you want to zip the root directory!?")
            else:
                archive_parent_dir = parent_dir
                file_name = target_dir

        # check for proper file name in the given directory
        archive_name = try_file_name(directory=archive_parent_dir,
                                  file_prefix=file_name,
                                  extension=archive_extension
                                  )
        if not archive_name.endswith('.' + archive_extension):
            archive_name += '.' + archive_extension
        archive_full_name = os.path.join(archive_parent_dir, archive_name)
        #
        if save_full_path:
            try:
                zip_handler = zipfile.ZipFile(archive_full_name, 'w')
            except(TypeError):
                print("\n *** Failed to Allow large files by setting allowZip64 to True...")
                print("Creating archive object with allowZip64 turned off...\n")
                zip_handler = zipfile.ZipFile(archive_full_name, 'w', allowZip64=allowZip64)

            for root, dirs, files in os.walk(path):
                for file in files:
                    zip_handler.write(os.path.join(root, file))
            zip_handler.close()
        else:
            cwd = os.getcwd()
            os.chdir(parent_dir)
            try:
                zip_handler = zipfile.ZipFile(archive_name, 'w', allowZip64=allowZip64)
            except(TypeError):
                zip_handler = zipfile.ZipFile(archive_name, 'w')
                print("\n *** Failed to Allow large files by setting allowZip64 to True...")
                print("Creating archive object with allowZip64 turned off...\n")

            for root, dirs, files in os.walk(target_dir):
                for file in files:
                    zip_handler.write(os.path.join(root, file))
            zip_handler.close()
            os.chdir(cwd)


# TODO: This method is not suitable for the new platform. Rewrite it in full. The model has to be involved of course...
def read_ensemble_states(ensemble_size,
                         ensemble_file_prefix='ensemble_mem',
                         ensemble_file_ext='.dat',
                         ensemble_relative_dir='ensemble_out/'
                         ):
    """
    Read ensemble states and return an np.ndarray (two-dimensional) with each ensemble member stored in a column.

    Args:
        ensemble_size: number of ensemble members to read from files
        ensemble_file_prefix: all files are named as <ensemble_file_prefix>_number.dat
        ensemble_relative_dir: relative directory containing the ensemble file(s)

    Returns:
        two-dimensional np.ndarray containing the ensemble members.

    """
    if not ensemble_file_ext.startswith('.'):
        ensemble_file_ext = '.'+ ensemble_file_ext

    raise NotImplementedError("To be written")

    return ensemble
