gatherDataIntoTable module
==========================

.. automodule:: gatherDataIntoTable
   :members:
   :undoc-members:
   :show-inheritance:

Dependencies
------------

Functions
---------

   .. function:: gatherDataIntoTable.buildTable(utteranceFiles,[,tableFileName='table'])

   This function reads the data from all files in the utteranceFiles list and writes all together into a HDF5 file.

   :param utteranceFiles: List with the name of the utterances that will be saved into the table.
   :type utteranceFiles: list
   :param tableFileName: Name of the file where the table is going to be stored.
   :type tableFileName: str, default='table'
   :rtype: None
   :return: This function does not return nothing to the program, but it creates a HDF5 file into the ``dirPath`` folder.

   Example of use::

      >>> import gatherDataIntoTable
      >>> uttType = 'whispered'
      >>> subset = 'test'
      >>> utteranceFiles = gatherDataIntoTable.getFilesList(uttType,subset=subset)
      >>> utteranceFiles
      ['emg_002-001-0222', 'emg_002-001-0248', 'emg_002-001-0219', 'emg_002-001-0216', 'emg_002-001-0206', 'emg_002-001-0237', 'emg_002-001-0214', 'emg_002-001-0234', 'emg_002-001-0205', 'emg_002-001-0201', 'emg_002-003-0238', 'emg_002-003-0208', 'emg_002-003-0222', 'emg_002-003-0224', 'emg_002-003-0246', 'emg_002-003-0233', 'emg_002-003-0205', 'emg_002-003-0244', 'emg_002-003-0225', 'emg_002-003-0207', 'emg_004-001-0117', 'emg_004-001-0116', 'emg_004-001-0140', 'emg_004-001-0138', 'emg_004-001-0118', 'emg_004-001-0149', 'emg_004-001-0107', 'emg_004-001-0129', 'emg_004-001-0100', 'emg_004-001-0148', 'emg_006-001-0117', 'emg_006-001-0116', 'emg_006-001-0140', 'emg_006-001-0138', 'emg_006-001-0118', 'emg_006-001-0149', 'emg_006-001-0107', 'emg_006-001-0129', 'emg_006-001-0100', 'emg_006-001-0148', 'emg_008-002-0118', 'emg_008-002-0117', 'emg_008-002-0141', 'emg_008-002-0139', 'emg_008-002-0119', 'emg_008-002-0150', 'emg_008-002-0108', 'emg_008-002-0130', 'emg_008-002-0101', 'emg_008-002-0149', 'emg_008-003-0131', 'emg_008-003-0144', 'emg_008-003-0103', 'emg_008-003-0128', 'emg_008-003-0102', 'emg_008-003-0130', 'emg_008-003-0116', 'emg_008-003-0112', 'emg_008-003-0111', 'emg_008-003-0125']
      >>> filename = 'newTable'
      >>> gatherDataIntoTable.buildTable(utteranceFiles, tableFileName=filename)

   This will create a file named ``newTable.h5`` in the ``EMG-UKA-Trial-Corpus`` folder, with the labels and the features of each frame in the selected utterances. In this case, the whispered utterances from every speakers and every sessions, both train and test subset.

   .. function:: gatherDataIntoTable.getFilesList(dirPath,uttType[,subset='both',speaker='all',session='all'])

   This function returns the list of all utterances of the given uttType and the given subset. Also a specific speaker and session can be selected.

   :param dirPath: Base path of the analyzed corpus.
   :type dirPath: str
   :param uttType: Type of utterance to be analyzed: ``audible``, ``whispered`` or ``silent``.
   :type uttType: str
   :param subset: Subset to be analyzed: ``train``, ``test`` or ``both``.
   :type subset: str, default='both'
   :param speaker: Speaker to be analyzed, or ``all`` if the files from every speaker are wanted to be returned.
   :type speaker: str, default='all'
   :param session: Session to be analyzed, or ``all`` if the files from every session are wanted to be returned. If the ``speakers`` parameter is ommited or set to ``all``, this parameter will be ignored and files from every session will be returned.
   :type session: str, default='all'
   :rtype: list
   :return: A list with the name of all the files belonging to the specified group.

   Example of use::

      >>> import gatherDataIntoTable
      >>> dirpath = 'EMG-UKA-Trial-Corpus'
      >>> uttType='whispered'
      >>> subset = 'test'
      >>> gatherDataIntoTable.getFilesList(dirpath,uttType)
      ['emg_002-001-0207', 'emg_002-001-0247', 'emg_002-001-0231', 'emg_002-001-0228', 'emg_002-001-0227', 'emg_002-001-0245', 'emg_002-001-0212', 'emg_002-001-0221', 'emg_002-001-0242', 'emg_002-001-0218', 'emg_002-001-0235', 'emg_002-001-0224', 'emg_002-001-0243', 'emg_002-001-0217', 'emg_002-001-0240', 'emg_002-001-0223', 'emg_002-001-0209', 'emg_002-001-0249', 'emg_002-001-0246', 'emg_002-001-0220', 'emg_002-001-0244', 'emg_002-001-0204', 'emg_002-001-0232', 'emg_002-001-0208', 'emg_002-001-0241', 'emg_002-001-0202', 'emg_002-001-0239', 'emg_002-001-0200', 'emg_002-001-0225', 'emg_002-001-0215', 'emg_002-001-0236', 'emg_002-001-0229', 'emg_002-001-0211', 'emg_002-001-0230', 'emg_002-001-0213', 'emg_002-001-0233', 'emg_002-001-0203', 'emg_002-001-0238', 'emg_002-001-0210', 'emg_002-001-0226', 'emg_002-003-0247', 'emg_002-003-0219', 'emg_002-003-0202', 'emg_002-003-0240', 'emg_002-003-0231', 'emg_002-003-0220', 'emg_002-003-0217', 'emg_002-003-0249', 'emg_002-003-0214', 'emg_002-003-0218', 'emg_002-003-0210', 'emg_002-003-0229', 'emg_002-003-0221', 'emg_002-003-0248', 'emg_002-003-0228', 'emg_002-003-0234', 'emg_002-003-0215', 'emg_002-003-0241', 'emg_002-003-0201', 'emg_002-003-0227', 'emg_002-003-0230', 'emg_002-003-0209', 'emg_002-003-0223', 'emg_002-003-0203', 'emg_002-003-0204', 'emg_002-003-0216', 'emg_002-003-0235', 'emg_002-003-0232', 'emg_002-003-0212', 'emg_002-003-0237', 'emg_002-003-0206', 'emg_002-003-0211', 'emg_002-003-0236', 'emg_002-003-0245', 'emg_002-003-0242', 'emg_002-003-0226', 'emg_002-003-0213', 'emg_002-003-0200', 'emg_002-003-0239', 'emg_002-003-0243', 'emg_004-001-0112', 'emg_004-001-0105', 'emg_004-001-0144', 'emg_004-001-0147', 'emg_004-001-0106', 'emg_004-001-0111', 'emg_004-001-0121', 'emg_004-001-0137', 'emg_004-001-0113', 'emg_004-001-0142', 'emg_004-001-0143', 'emg_004-001-0139', 'emg_004-001-0103', 'emg_004-001-0134', 'emg_004-001-0131', 'emg_004-001-0136', 'emg_004-001-0141', 'emg_004-001-0126', 'emg_004-001-0120', 'emg_004-001-0110', 'emg_004-001-0146', 'emg_004-001-0135', 'emg_004-001-0130', 'emg_004-001-0102', 'emg_004-001-0119', 'emg_004-001-0132', 'emg_004-001-0133', 'emg_004-001-0128', 'emg_004-001-0122', 'emg_004-001-0101', 'emg_004-001-0109', 'emg_004-001-0104', 'emg_004-001-0125', 'emg_004-001-0108', 'emg_004-001-0114', 'emg_004-001-0123', 'emg_004-001-0115', 'emg_004-001-0124', 'emg_004-001-0127', 'emg_004-001-0145', 'emg_006-001-0112', 'emg_006-001-0105', 'emg_006-001-0144', 'emg_006-001-0147', 'emg_006-001-0106', 'emg_006-001-0111', 'emg_006-001-0121', 'emg_006-001-0137', 'emg_006-001-0113', 'emg_006-001-0142', 'emg_006-001-0143', 'emg_006-001-0139', 'emg_006-001-0103', 'emg_006-001-0134', 'emg_006-001-0131', 'emg_006-001-0136', 'emg_006-001-0141', 'emg_006-001-0126', 'emg_006-001-0120', 'emg_006-001-0110', 'emg_006-001-0146', 'emg_006-001-0135', 'emg_006-001-0130', 'emg_006-001-0102', 'emg_006-001-0119', 'emg_006-001-0132', 'emg_006-001-0133', 'emg_006-001-0128', 'emg_006-001-0122', 'emg_006-001-0101', 'emg_006-001-0109', 'emg_006-001-0104', 'emg_006-001-0125', 'emg_006-001-0108', 'emg_006-001-0114', 'emg_006-001-0123', 'emg_006-001-0115', 'emg_006-001-0124', 'emg_006-001-0127', 'emg_006-001-0145', 'emg_008-002-0113', 'emg_008-002-0106', 'emg_008-002-0145', 'emg_008-002-0148', 'emg_008-002-0107', 'emg_008-002-0112', 'emg_008-002-0122', 'emg_008-002-0138', 'emg_008-002-0114', 'emg_008-002-0143', 'emg_008-002-0144', 'emg_008-002-0140', 'emg_008-002-0104', 'emg_008-002-0135', 'emg_008-002-0132', 'emg_008-002-0137', 'emg_008-002-0142', 'emg_008-002-0127', 'emg_008-002-0121', 'emg_008-002-0111', 'emg_008-002-0147', 'emg_008-002-0136', 'emg_008-002-0131', 'emg_008-002-0103', 'emg_008-002-0120', 'emg_008-002-0133', 'emg_008-002-0134', 'emg_008-002-0129', 'emg_008-002-0123', 'emg_008-002-0102', 'emg_008-002-0110', 'emg_008-002-0105', 'emg_008-002-0126', 'emg_008-002-0109', 'emg_008-002-0115', 'emg_008-002-0124', 'emg_008-002-0116', 'emg_008-002-0125', 'emg_008-002-0128', 'emg_008-002-0146', 'emg_008-003-0149', 'emg_008-003-0141', 'emg_008-003-0136', 'emg_008-003-0139', 'emg_008-003-0121', 'emg_008-003-0105', 'emg_008-003-0140', 'emg_008-003-0147', 'emg_008-003-0106', 'emg_008-003-0145', 'emg_008-003-0110', 'emg_008-003-0114', 'emg_008-003-0126', 'emg_008-003-0137', 'emg_008-003-0133', 'emg_008-003-0101', 'emg_008-003-0107', 'emg_008-003-0142', 'emg_008-003-0132', 'emg_008-003-0124', 'emg_008-003-0123', 'emg_008-003-0138', 'emg_008-003-0108', 'emg_008-003-0143', 'emg_008-003-0122', 'emg_008-003-0109', 'emg_008-003-0104', 'emg_008-003-0135', 'emg_008-003-0113', 'emg_008-003-0115', 'emg_008-003-0120', 'emg_008-003-0129', 'emg_008-003-0134', 'emg_008-003-0146', 'emg_008-003-0118', 'emg_008-003-0148', 'emg_008-003-0150', 'emg_008-003-0119', 'emg_008-003-0127', 'emg_008-003-0117', 'emg_002-001-0222', 'emg_002-001-0248', 'emg_002-001-0219', 'emg_002-001-0216', 'emg_002-001-0206', 'emg_002-001-0237', 'emg_002-001-0214', 'emg_002-001-0234', 'emg_002-001-0205', 'emg_002-001-0201', 'emg_002-003-0238', 'emg_002-003-0208', 'emg_002-003-0222', 'emg_002-003-0224', 'emg_002-003-0246', 'emg_002-003-0233', 'emg_002-003-0205', 'emg_002-003-0244', 'emg_002-003-0225', 'emg_002-003-0207', 'emg_004-001-0117', 'emg_004-001-0116', 'emg_004-001-0140', 'emg_004-001-0138', 'emg_004-001-0118', 'emg_004-001-0149', 'emg_004-001-0107', 'emg_004-001-0129', 'emg_004-001-0100', 'emg_004-001-0148', 'emg_006-001-0117', 'emg_006-001-0116', 'emg_006-001-0140', 'emg_006-001-0138', 'emg_006-001-0118', 'emg_006-001-0149', 'emg_006-001-0107', 'emg_006-001-0129', 'emg_006-001-0100', 'emg_006-001-0148', 'emg_008-002-0118', 'emg_008-002-0117', 'emg_008-002-0141', 'emg_008-002-0139', 'emg_008-002-0119', 'emg_008-002-0150', 'emg_008-002-0108', 'emg_008-002-0130', 'emg_008-002-0101', 'emg_008-002-0149', 'emg_008-003-0131', 'emg_008-003-0144', 'emg_008-003-0103', 'emg_008-003-0128', 'emg_008-003-0102', 'emg_008-003-0130', 'emg_008-003-0116', 'emg_008-003-0112', 'emg_008-003-0111', 'emg_008-003-0125']
      >>> gatherDataIntoTable.getFilesList(dirpath,uttType,subset=subset)
      ['emg_002-001-0222', 'emg_002-001-0248', 'emg_002-001-0219', 'emg_002-001-0216', 'emg_002-001-0206', 'emg_002-001-0237', 'emg_002-001-0214', 'emg_002-001-0234', 'emg_002-001-0205', 'emg_002-001-0201', 'emg_002-003-0238', 'emg_002-003-0208', 'emg_002-003-0222', 'emg_002-003-0224', 'emg_002-003-0246', 'emg_002-003-0233', 'emg_002-003-0205', 'emg_002-003-0244', 'emg_002-003-0225', 'emg_002-003-0207', 'emg_004-001-0117', 'emg_004-001-0116', 'emg_004-001-0140', 'emg_004-001-0138', 'emg_004-001-0118', 'emg_004-001-0149', 'emg_004-001-0107', 'emg_004-001-0129', 'emg_004-001-0100', 'emg_004-001-0148', 'emg_006-001-0117', 'emg_006-001-0116', 'emg_006-001-0140', 'emg_006-001-0138', 'emg_006-001-0118', 'emg_006-001-0149', 'emg_006-001-0107', 'emg_006-001-0129', 'emg_006-001-0100', 'emg_006-001-0148', 'emg_008-002-0118', 'emg_008-002-0117', 'emg_008-002-0141', 'emg_008-002-0139', 'emg_008-002-0119', 'emg_008-002-0150', 'emg_008-002-0108', 'emg_008-002-0130', 'emg_008-002-0101', 'emg_008-002-0149', 'emg_008-003-0131', 'emg_008-003-0144', 'emg_008-003-0103', 'emg_008-003-0128', 'emg_008-003-0102', 'emg_008-003-0130', 'emg_008-003-0116', 'emg_008-003-0112', 'emg_008-003-0111', 'emg_008-003-0125']
      >>> gatherDataIntoTable.getFilesList(dirpath,uttType,subset=subset,speaker='002')
      ['emg_002-001-0222', 'emg_002-001-0248', 'emg_002-001-0219', 'emg_002-001-0216', 'emg_002-001-0206', 'emg_002-001-0237', 'emg_002-001-0214', 'emg_002-001-0234', 'emg_002-001-0205', 'emg_002-001-0201', 'emg_002-003-0238', 'emg_002-003-0208', 'emg_002-003-0222', 'emg_002-003-0224', 'emg_002-003-0246', 'emg_002-003-0233', 'emg_002-003-0205', 'emg_002-003-0244', 'emg_002-003-0225', 'emg_002-003-0207']
      >>> gatherDataIntoTable.getFilesList(dirpath,uttType,subset=subset,speaker='002',session='001')
      ['emg_002-001-0222', 'emg_002-001-0248', 'emg_002-001-0219', 'emg_002-001-0216', 'emg_002-001-0206', 'emg_002-001-0237', 'emg_002-001-0214', 'emg_002-001-0234', 'emg_002-001-0205', 'emg_002-001-0201']

   .. function:: gatherDataIntoTable.main(dirPath,uttType[,subset='both',speaker='all',session='all'])

      This is the main function of the :mod:`gatherDataIntoTable`. It searches for the specified utterances, extracts the features and the labels from their frames and puts them together into a HDF5 file.

      :param dirPath: Base path of the analyzed corpus.
      :type dirPath: str
      :param uttType: Type of utterance to be analyzed: ``audible``, ``whispered`` or ``silent``.
      :type uttType: str
      :param subset: Subset to be analyzed: ``train``, ``test`` or ``both``.
      :type subset: str, default='both'
      :param speaker: Speaker to be analyzed, or ``all`` if the files from every speaker are wanted to be returned.
      :type speaker: str, default='all'
      :param session: Session to be analyzed, or ``all`` if the files from every session are wanted to be returned. If the ``speakers`` parameter is ommited or set to ``all``, this parameter will be ignored and files from every session will be returned.
      :type session: str, default='all'
      :rtype: None
      :return: This function does not return nothing to the program, but it creates a HDF5 file into the ``dirPath`` folder.

      Example of use::

         >>> import gatherDataIntoTable
         >>> dirpath = 'EMG-UKA-Trial-Corpus'
         >>> uttType='whispered'
         >>> subset = 'test'
         >>> speaker = '002'
         >>> session = '001'
         >>> gatherDataIntoTable.main(dirpath,uttType,subset=subset,speaker=speaker,session=session)

      This will create a file named ``002_001_whispered_test_Table.h5`` into the ``EMG-UKA-Trial-Corpus`` folder, with the frames from the whispered utterances of the test file from the session 001 of the speaker 002.