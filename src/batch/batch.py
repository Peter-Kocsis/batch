import copy
import fnmatch
from collections import OrderedDict
from typing import Mapping, Union

from .utils import split_list


class Batch(Mapping):
    """
    Generic class implementing a batch as a dictionary of objects.
    It supports every member functions of the underlying objects.
    """
    INTERNAL_MEMBER_PREFIX = "_Batch"

    """ ====================================== INSTANTIATE ====================================== """

    def __init__(self, *args, default=None, **kwargs):
        """
        Constructing a new batch.
        :param args: Batch elements as dictionary
        :param default: Default value constructor if a key is not found. Otherwise a KeyError is raised.
        :param kwargs: Batch elements as keyword arguments
        """
        super().__init__()
        self.__default = default
        self.__dict__.update(**kwargs)
        self.__in_place = False

        # Batch elements as dictionary
        if len(args) == 1 and isinstance(args[0], dict):
            self.__dict__.update(**args[0])

        # Set the operation functions
        def make_func(name, in_place):
            return lambda caller, *args: caller._get_member_attribute(name=name, in_place=in_place)(*args)

        for name in [
            # Unary operators
            "__not__", "__abs__", "__index__", "__inv__", "__invert__", "__neg__", "__pos__",

            # Binary operators
            "__add__", "__and__", "__concat__", "__floordiv__", "__lshift__", "__mod__", "__mul__",
            "__or__", "__pow__", "__rshift__", "__sub__", "__truediv__", "__xor__", "__eq__",

            # Reverse operators
            "__radd__", "__rand__", "__rmul__",
            "__ror__", "__rsub__", "__rxor__"]:
            setattr(Batch, name, make_func(name, in_place=False))

        for name in [
            # In-place operators
            "__iadd__", "__iand__", "__iconcat__", "__ifloordiv__", "__ilshift__", "__imod__", "__imul__",
            "__ior__", "__ipow__", "__irshift__", "__isub__", "__itruediv__", "__ixor__"]:
            setattr(Batch, name, make_func(name, in_place=True))

        # Batchify recursively
        self._batchify()

    @classmethod
    def from_dict(cls, data):
        """
        Constructing a new batch from a dictionary
        :param data: Dictionary containing the data
        :return: Created batch
        """
        other = cls()
        for key, value in data.items():
            if isinstance(value, (dict, Batch)):
                other[key] = cls.from_dict(value)
            else:
                other[key] = value
        return other

    @classmethod
    def from_tensor(cls, data, cat_map, dim=0, split_fn=split_list):
        """
        Constructing a new batch from a tensor
        :param data: Tensor containing the data
        :param cat_map: Dictionary of the member names and their sizes
        :param dim: Dimension along which to split the tensor
        :param split_fn: Split function
        :return: Created batch
        """
        other = cls()
        for key, value in cat_map.items():
            if isinstance(value, int):
                other[key], data = split_fn(data, [value, data.shape[dim] - value], dim)
            if isinstance(value, OrderedDict):
                other[key] = cls.from_tensor(data, value, dim=dim, split_fn=split_fn)
        return other

    @classmethod
    def from_batch_list(cls, *args):
        """
        Constructing a new batch with the list of members from a list of batches
        :param args: Batches to be merged
        :return: Created batch
        """
        other = cls()

        batch_args = [arg for arg in list(args)]
        all_keys = {key for batch_arg in batch_args for key in batch_arg.keys()}

        for key in all_keys:
            other.__dict__[key] = [batch_arg[key] for batch_arg in batch_args if key in batch_arg]

        # Convert all member batch lists as well
        for key in other.keys():
            if isinstance(other[key][0], Batch):
                other[key] = cls.from_batch_list(*other[key])
        return other

    def copy(self, deep=True):
        """
        Creates a deep copy of the batch
        :return: The created copy
        """
        if deep:
            return copy.deepcopy(self)
        else:
            return copy.copy(self)

    def __deepcopy__(self, memo=None):
        other = Batch()
        for key in self.keys():
            other[key] = copy.deepcopy(self[key], memo=memo)
        return other

    """ ====================================== PROCESSING ====================================== """

    def map(self, fn, *args, **kwargs):
        """
        Applies a function to all members of the batch
        :param fn: Function to be applied
        :return: Batch of the results
        """
        other = Batch()
        for key in self.keys():
            if isinstance(self[key], Batch):
                other[key] = self[key].map(fn, *args, **kwargs)
            else:
                other[key] = fn(self[key], *args, **kwargs)
        return other

    def map_keys(self, fn, *args, **kwargs):
        """
        Applies a function to all keys of the batch
        :param fn: Function to be applied
        :return: Batch of the results
        """
        other = Batch()
        for key in self.keys():
            if isinstance(self[key], Batch):
                other[fn(key, *args, **kwargs)] = self[key].map_keys(fn, *args, **kwargs)
            else:
                other[fn(key, *args, **kwargs)] = self[key]
        return other

    def filter(self, fn, *args, **kwargs):
        """
        Filters all members of the batch
        :param fn: Predicate function
        :return: Batch of the results
        """
        other = Batch()
        for key in self.keys():
            if isinstance(self[key], Batch):
                other[key] = self[key].filter(fn, *args, **kwargs)
            else:
                if fn(self[key], *args, **kwargs):
                    other[key] = self[key]
        return other

    def flatten(self, separator="."):
        """
        Flattens the batch into a single batch
        :return: The flattened batch
        """
        other = Batch()
        for key in self.keys():
            if isinstance(self[key], Batch):
                other.update(self[key].flatten(separator=separator).add_prefix(key + separator))
            else:
                other[key] = self[key]
        return other

    def add_prefix(self, prefix):
        """
        Adds a prefix to all keys
        :param prefix: Prefix to be added
        :return: The prefixed batch
        """
        other = Batch()
        for key in self.keys():
            other[prefix + key] = self[key]
        return other

    def add_postfix(self, postfix):
        """
        Adds a prefix to all keys
        :param prefix: Prefix to be added
        :return: The prefixed batch
        """
        other = Batch()
        for key in self.keys():
            other[key + postfix] = self[key]
        return other

    def remap(self, mapping):
        """
        Remaps the keys of the batch
        :param mapping: Dictionary of the old and new keys
        :return: The remapped batch
        """
        other = Batch()
        for old_key, new_key in mapping.items():
            assert isinstance(old_key, str), "Only str remapping is allowed!"
            assert isinstance(new_key, str), "Only str remapping is allowed!"
            if old_key in self:
                other[new_key] = self[old_key]
            else:
                other[new_key] = None
        return other

    def transpose(self):
        """
        Transposes the batch
        :return: The transposed batch
        """
        other = Batch()
        for key, value in self.items():
            assert isinstance(value, str), "Only str batches can be transposed!"
            assert value not in other, f"Cannot transpose batch with duplicate keys ({value})!"
            other[value] = key
        return other

    """ ====================================== CONVERT ====================================== """

    def to_dict(self):
        """
        Converts the batch into a dictionary
        :return: The dictionary
        """
        other = dict()
        for key in self.keys():
            if isinstance(self[key], Batch):
                other[key] = self[key].to_dict()
            else:
                other[key] = self[key]
        return other

    def to_list(self):
        idx = 0
        elements = []
        try:
            while True:
                next_element = self[idx]
                if len(next_element) == 0:
                    break
                elements.append(next_element)
                idx += 1
        except IndexError:
            pass
        return elements

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"Batch({list(self.items())})"

    """ ====================================== MEMBER ACCESS ====================================== """

    def __contains__(self, key):
        try:
            return len(self[key]) > 0
        except KeyError:
            return False

    def __getitem__(self, index_or_key):
        """
        Accessing a member of the batch.
        :param index_or_key: Index (int|tuple|list) or key (str) of the member.
          - If a key is given, then the member of the batch is returned.
          - If an index is given, then a new batch is returned containing the indexed elements.
          - If a tuple|list of strings are given, a new batch is created with those keys
        :return: The extracted member.
        """
        if index_or_key is None:
            return self
        elif isinstance(index_or_key, str):
            return self._getitem_key(key=index_or_key)
        elif isinstance(index_or_key, int):
            return self._getitem_index(index=index_or_key)
        elif isinstance(index_or_key, (tuple, list)):
            if len(index_or_key) == 0:
                return Batch()
            if all(type(index_or_key[0]) == type(idx) for idx in index_or_key):
                if isinstance(index_or_key[0], int):
                    return self._getitem_index(index=index_or_key)
                elif isinstance(index_or_key[0], str):
                    other = Batch()
                    for key in index_or_key:
                        other[key] = self._getitem_key(key=key)
                    return other
                else:
                    raise NotImplementedError(f"Index type {type(index_or_key[0])} in {type(index_or_key)} not supported")
            else: # If not all indices are of the same type, then we assume that the indices are slices or integers
                assert all(isinstance(idx, (int, slice)) for idx in index_or_key), "Only slices and integers are supported"
                return self._getitem_index(index=index_or_key)
        else:
            raise NotImplementedError(f"Index type {type(index_or_key)} not supported")

    def _getitem_key(self, key: str):
        """
        Gets a member by its name
        :param key: String key
        :return: Extracted value
        """
        # Check if key in the dictionary
        if key in self.__dict__:
            return self.__dict__.__getitem__(key)

        # Check if the key refers to a subvalue separated by '.' character.
        if "." in key:
            root_key, sub_key = key.split(".", maxsplit=1)
            try:
                return self._getitem_key(key=root_key)[sub_key]
            except KeyError:
                pass

        # Check if default constructor is given
        if self.__default is not None:
            self.__dict__.__setitem__(key, self.__default())
            return self.__dict__.__getitem__(key)

        raise KeyError(f"Key {key} not found in {list(self.keys())}")

    def _getitem_index(self, index: Union[int, tuple, list]) -> "Batch":
        """
        Indexes the members of the batch
        :param index: Index at the batch
        :return: Extracted value
        """
        other = Batch()
        for key in self.keys():
            other.__dict__[key] = self.__dict__[key][index]
        return other

    def query_wildcard(self, query):
        """
        Query the batch with a wildcard query.
        :param query: Query string
        :return: Batch of the results
        """
        if isinstance(query, str):
            query = [query]

        queried_keys = [k for q in query for k in fnmatch.filter(self.keys(recursive=True), q)]
        return self[queried_keys]

    def __setitem__(self, index_or_key, value):
        """
        Setting a member of the batch.
        :param index_or_key: Index (int|tuple|list) or key (str) of the member.
          - If a key is given, then the member of the batch is set.
          - If an index is given, then a new batch is set containing the indexed elements.
          - If a tuple|list of strings are given, those keys will be set
        :return: The modified object
        """
        if isinstance(index_or_key, str):
            return self._setitem_key(key=index_or_key, value=value)
        elif isinstance(index_or_key, int):
            return self._setitem_index(index=index_or_key, value=value)
        elif isinstance(index_or_key, (tuple, list)):
            assert all(type(index_or_key[0]) == type(idx) for idx in
                       index_or_key), f"All indices must be of the same type, but are {index_or_key}"
            if isinstance(index_or_key[0], int):
                return self._setitem_index(index=index_or_key, value=value)
            elif isinstance(index_or_key[0], str):
                assert isinstance(value, Batch), "Value must be a batch if an index is given"
                for key in index_or_key:
                    self._setitem_key(key=key, value=value._getitem_key(key=key))
                return self
            else:
                raise NotImplementedError(f"Index type {type(index_or_key[0])} in {type(index_or_key)} not supported")
        raise NotImplementedError(f"Index type {type(index_or_key)} not supported")

    def _setitem_key(self, key: str, value):
        """
        Sets a member by its name
        :param key: String key
        :param value: Value to set
        :return: Extracted value
        """
        # Check if the key refers to a subvalue separated by '.' character.
        if "." in key:
            root_key, sub_key = key.split(".", maxsplit=1)
            try:
                sub_item = self[root_key]
                if isinstance(sub_item, Batch):
                    return sub_item._setitem_key(key=sub_key, value=value)
            except KeyError:
                pass

        self.__dict__.__setitem__(key, value)
        return self

    def _setitem_index(self, index: Union[int, tuple, list], value):
        """
        Indexes the members of the batch
        :param index: Index at the batch
        :param value: Value to set
        :return: Extracted value
        """
        assert isinstance(value, Batch), "Value must be a batch if an index is given"
        for key in self.keys():
            self.__dict__[key] = value.__dict__[key][index]
        return self

    def __delitem__(self, k):
        assert isinstance(k, str), "Only string keys are supported"
        self.__dict__.__delitem__(k)

    def __len__(self) -> int:
        return len(list(self.keys()))

    """ ====================================== DICT METHODS ====================================== """

    def keys(self, depth=0):
        assert depth >= -1, "Depth must be greater or equal to -1"
        if depth == 0:
            return (key for key in self.__dict__.keys() if not key.startswith(self.INTERNAL_MEMBER_PREFIX))
        else:
            keys = []
            for key in self.__dict__.keys():
                if not key.startswith(self.INTERNAL_MEMBER_PREFIX):
                    if isinstance(self.__dict__[key], Batch):
                        keys += [f"{key}.{sub_key}"
                                 for sub_key in self.__dict__[key].keys(depth=depth - 1 if depth > 0 else -1)]
                    else:
                        keys.append(key)
            return keys

    def __iter__(self):
        for key in self.keys():
            if not key.startswith(self.INTERNAL_MEMBER_PREFIX):
                yield key

    def items(self):
        for key in self.keys():
            yield key, self.__dict__[key]

    def update(self, *args, **kwargs):
        for arg in args:
            if isinstance(arg, dict):
                self.__dict__.update(arg)
            elif isinstance(arg, Batch):
                self.__dict__.update(arg.__dict__)
            else:
                raise NotImplementedError(f"Update not implemented for {type(arg)}")
        self.__dict__.update(kwargs)
        return self

    def pop(self, index):
        return self.__dict__.pop(index)

    def __getstate__(self):
        """
        Serializes the batch
        :return:
        """
        return self.__dict__

    def __setstate__(self, d):
        """
        Deserializes the batch
        :param d:
        :return:
        """
        self.__dict__ = d

    """ ====================================== INTERNAL PROCESSING ====================================== """

    def _batchify(self):
        """
        Converts all members into batches
        :return: The batchified batch
        """
        for key in self.keys():
            if isinstance(self[key], dict):
                self[key] = Batch(self[key])
        return self

    """ ====================================== OPERATIONS ====================================== """

    def __getattribute__(self, name):
        """
        Generic method called on every attribute query.
        If the attribute is not defined, then a function call is assumed and a function pointer is returned,
        which applies the function to all members.
        :param name: Attribute to be queried
        :return: Function or attribute value
        """
        try:
            return super().__getattribute__(name)
        except AttributeError:
            return self._get_member_attribute(name)

    def _get_member_attribute(self, name, in_place=False):
        """
        Creates a new batch with the member attributes.
        :param name: Attribute to be queried
        :return: Batch of the member attributes
        """
        if len(self) == 0:
            raise AttributeError("Cannot get member function of empty batch")

        if in_place:
            other = self
        else:
            other = Batch()

        other.__in_place = in_place

        for key in self.keys():
            attr = self.__dict__[key]
            if hasattr(attr, name):
                other.__dict__[key] = getattr(attr, name)
            else:
                raise AttributeError(f"Member function {name} not implemented for {key} - {type(attr)}")
        return other

    def __call__(self, *args, **kwargs):
        """
        Calls all members with arguments
        :param args:
        :param in_place:
        :param kwargs:
        :return:
        """
        if self.__in_place:
            other = self
        else:
            other = Batch()

        for key in self.keys():
            args_for_key = [arg if not isinstance(arg, Batch) else arg[key] for arg in args]
            kwargs_for_key = {k: val if not isinstance(val, Batch) else val[key] for k, val in kwargs.items()}
            other.__dict__[key] = self.__dict__[key](*args_for_key, **kwargs_for_key)
        return other
