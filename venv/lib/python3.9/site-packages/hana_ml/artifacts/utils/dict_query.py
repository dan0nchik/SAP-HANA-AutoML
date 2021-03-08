"""
This module provides dictionary helper functionality.
"""
#pylint: disable=not-an-iterable
class DictUtils(dict):
    """
    This class provides dictionary helper functionality.
    """
    @staticmethod
    def get(path, default=None):
        """
        Generically retrieve a value from a key which can be multiple levels
        deep.

        Parameters
        ----------
        path : str
            Path to the value (mutliple levels)
        default : str
            The value that needs to be returned if the key does not exist
        """
        keys = path.split("/")
        val = None

        for key in keys:
            if val:
                if isinstance(val, list):
                    val = [v.get(key, default) if v else None for v in val]
                else:
                    val = val.get(key, default)
            else:
                val = dict.get(key, default)

            if not val:
                break

        return val
