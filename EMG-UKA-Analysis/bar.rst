bar module
==========

.. automodule:: bar
   :members:
   :undoc-members:
   :show-inheritance:

Functions
---------

.. function:: bar.printProgressBar(iteration, total[, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"])

   Call in a loop to create terminal progress bar.

   .. note::
      This function has been copied from https://gist.github.com/greenstick/b23e475d2bfdc3a82e34eaa1f6781ee4.

   :param iteration: Current iteration.
   :type iteration: int
   :param total: Total iterations.
   :type total: int
   :param prefix: Prefix string.
   :type prefix: str, default=''
   :param suffix: Suffix string.
   :type suffix: str, default=''
   :param decimals: Positive number of decimals in percent complete.
   :type decimals: int, default=1
   :param length: Character length of bar.
   :type length: int, default=100
   :param fill: Bar fill character.
   :type fill: str, default='█'
   :param printEnd: End character (e.g. "\r" "\r\n")
   :type printEnd: str, default='\r'

   Example of use:

      >>> import bar
      >>> totalIterations = 4
      >>> for i in range(0,totalIterations):
      ...     bar.printProgressBar(i, totalIterations-1, prefix='Progress: ', suffix='completed', length=50, printEnd = "\r\n")
      ...
      Progress:  |--------------------------------------------------| 0.0% completed
      Progress:  |████████████████----------------------------------| 33.3% completed
      Progress:  |█████████████████████████████████-----------------| 66.7% completed
      Progress:  |██████████████████████████████████████████████████| 100.0% completed