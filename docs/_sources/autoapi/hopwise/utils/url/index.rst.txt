hopwise.utils.url
=================

.. py:module:: hopwise.utils.url

.. autoapi-nested-parse::

   hopwise.utils.url
   ################################
   Reference code:
       https://github.com/snap-stanford/ogb/blob/master/ogb/utils/url.py



Attributes
----------

.. autoapisummary::

   hopwise.utils.url.GBFACTOR


Functions
---------

.. autoapisummary::

   hopwise.utils.url.decide_download
   hopwise.utils.url.makedirs
   hopwise.utils.url.download_url
   hopwise.utils.url.extract_zip
   hopwise.utils.url.rename_atomic_files


Module Contents
---------------

.. py:data:: GBFACTOR

.. py:function:: decide_download(url)

.. py:function:: makedirs(path)

.. py:function:: download_url(url, folder)

   Downloads the content of an URL to a specific folder.

   :param url: The url.
   :type url: string
   :param folder: The folder.
   :type folder: string


.. py:function:: extract_zip(path, folder)

   Extracts a zip archive to a specific folder.

   :param path: The path to the tar archive.
   :type path: string
   :param folder: The folder.
   :type folder: string


.. py:function:: rename_atomic_files(folder, old_name, new_name)

   Rename all atomic files in a given folder.

   :param folder: The folder.
   :type folder: string
   :param old_name: Old name for atomic files.
   :type old_name: string
   :param new_name: New name for atomic files.
   :type new_name: string


