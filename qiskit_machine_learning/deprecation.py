# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Contains the Deprecation message methods."""

from abc import abstractmethod
import warnings
import functools
import inspect
from typing import NamedTuple, Optional, Callable, Dict, Set, cast, Any
from enum import Enum, EnumMeta


class DeprecatedEnum(Enum):
    """
    Shows deprecate message whenever member is accessed
    """

    def __new__(cls, value, *args):
        member = object.__new__(cls)
        member._value_ = value
        member._args = args
        member._show_deprecate = member.deprecate
        return member

    @abstractmethod
    def deprecate(self):
        """show deprecate message"""
        pass


class DeprecatedEnumMeta(EnumMeta):
    """
    Shows deprecate message whenever member is accessed
    """

    def __getattribute__(cls, name):
        obj = super().__getattribute__(name)
        if isinstance(obj, DeprecatedEnum) and obj._show_deprecate:
            obj._show_deprecate()
        return obj

    def __getitem__(cls, name):
        member = super().__getitem__(name)
        if member._show_deprecate:
            member._show_deprecate()
        return member

    # pylint: disable=redefined-builtin
    def __call__(cls, value, names=None, *, module=None, qualname=None, type=None, start=1):
        obj = super().__call__(
            value, names, module=module, qualname=qualname, type=type, start=start
        )
        if isinstance(obj, DeprecatedEnum) and obj._show_deprecate:
            obj._show_deprecate()
        return obj


class DeprecatedType(Enum):
    """ " Deprecation Types"""

    PACKAGE = "package"
    ENUM = "enum"
    CLASS = "class"
    METHOD = "method"
    FUNCTION = "function"
    ARGUMENT = "argument"
    PROPERTY = "property"


class _DeprecatedTypeName(NamedTuple):
    version: str
    old_type: DeprecatedType
    old_name: str
    new_type: DeprecatedType
    new_name: str
    additional_msg: str


class _DeprecatedArgument(NamedTuple):
    version: str
    func_qualname: str
    old_arg: str
    new_arg: str
    additional_msg: str


class _DeprecatedValue(NamedTuple):
    version: str
    func_qualname: str
    argument: str
    old_value: str
    new_value: str
    additional_msg: str


_DEPRECATED_OBJECTS: Set[NamedTuple] = set()


def warn_deprecated(
    version: str,
    old_type: DeprecatedType,
    old_name: str,
    new_type: Optional[DeprecatedType] = None,
    new_name: Optional[str] = None,
    additional_msg: Optional[str] = None,
    stack_level: int = 2,
) -> None:
    """Emits deprecation warning the first time only
    Args:
        version: Version to be used
        old_type: Old type to be used
        old_name: Old name to be used
        new_type: New type to be used, if None, old_type is used instead.
        new_name: New name to be used
        additional_msg: any additional message
        stack_level: stack level
    """
    # skip if it was already added
    obj = _DeprecatedTypeName(version, old_type, old_name, new_type, new_name, additional_msg)
    if obj in _DEPRECATED_OBJECTS:
        return

    _DEPRECATED_OBJECTS.add(cast(NamedTuple, obj))

    msg = (
        f"The {old_name} {old_type.value} is deprecated as of version {version} "
        "and will be removed no sooner than 3 months after the release"
    )
    if new_name:
        type_str = new_type.value if new_type is not None else old_type.value
        msg += f". Instead use the {new_name} {type_str}"
    if additional_msg:
        msg += f" {additional_msg}"
    msg += "."

    warnings.warn(msg, DeprecationWarning, stacklevel=stack_level + 1)


def warn_deprecated_same_type_name(
    version: str,
    new_type: DeprecatedType,
    new_name: str,
    additional_msg: Optional[str] = None,
    stack_level: int = 2,
) -> None:
    """Emits deprecation warning the first time only
       Used when the type and name remained the same.
    Args:
        version: Version to be used
        new_type: new type to be used
        new_name: new name to be used
        additional_msg: any additional message
        stack_level: stack level
    """
    warn_deprecated(
        version, new_type, new_name, new_type, new_name, additional_msg, stack_level + 1
    )


def _rename_kwargs(version, qualname, func_name, kwargs, kwarg_map, additional_msg, stack_level):
    for old_arg, new_arg in kwarg_map.items():
        if old_arg in kwargs:
            if new_arg in kwargs:
                raise TypeError(
                    "{} received both {} and {} (deprecated).".format(func_name, new_arg, old_arg)
                )

            # skip if it was already added
            obj = _DeprecatedArgument(version, qualname, old_arg, new_arg, additional_msg)
            if obj not in _DEPRECATED_OBJECTS:
                _DEPRECATED_OBJECTS.add(obj)

                msg = (
                    f"{func_name}: the {old_arg} {DeprecatedType.ARGUMENT.value} is deprecated "
                    f"as of version {version} and will be removed no sooner "
                    "than 3 months after the release. Instead use the "
                    f"{new_arg} {DeprecatedType.ARGUMENT.value}"
                )
                if additional_msg:
                    msg += f" {additional_msg}"
                msg += "."
                warnings.warn(msg, DeprecationWarning, stacklevel=stack_level)
            kwargs[new_arg] = kwargs.pop(old_arg)


def deprecate_arguments(
    version: str,
    kwarg_map: Dict[str, str],
    additional_msg: Optional[str] = None,
    stack_level: int = 3,
) -> Callable:
    """Decorator to alias deprecated argument names and warn upon use.
    Args:
        version: Version to be used
        kwarg_map: Args dictionary with old, new arguments
        additional_msg: any additional message
        stack_level: stack level

    Returns:
        The decorated function
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if kwargs:
                _rename_kwargs(
                    version,
                    func.__qualname__,
                    func.__name__,
                    kwargs,
                    kwarg_map,
                    additional_msg,
                    stack_level,
                )
            return func(*args, **kwargs)

        return wrapper

    return decorator


def _check_values(version, qualname, args, kwargs, kwarg_map, additional_msg, stack_level):
    for arg, value_dict in kwarg_map.items():
        index = value_dict["index"]
        values_map = value_dict["values"]
        old_value = None
        if args and 0 < index < len(args):
            old_value = args[index]
        if kwargs and arg in kwargs:
            old_value = kwargs[arg]

        if old_value in values_map:
            new_value = values_map[old_value]

            # skip if it was already added
            obj = _DeprecatedValue(version, qualname, arg, old_value, new_value, additional_msg)
            if obj in _DEPRECATED_OBJECTS:
                continue

            _DEPRECATED_OBJECTS.add(obj)

            msg = (
                f'The {arg} {DeprecatedType.ARGUMENT.value} value "{old_value}" is deprecated '
                f"as of version {version} and will be removed no sooner "
                "than 3 months after the release. Instead use the "
                f'"{new_value}" value'
            )
            if additional_msg:
                msg += f" {additional_msg}"
            msg += "."
            warnings.warn(msg, DeprecationWarning, stacklevel=stack_level)


def deprecate_values(
    version: str,
    kwarg_map: Dict[str, Dict[Any, Any]],
    additional_msg: Optional[str] = None,
    stack_level: int = 3,
) -> Callable:
    """Decorator to alias deprecated default values and warn upon use.
    Args:
        version: Version to be used
        kwarg_map: Args dictionary with argument and map of old, new values
        additional_msg: any additional message
        stack_level: stack level

    Returns:
        The decorated function
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if args or kwargs:
                parameter_names = list(inspect.signature(func).parameters.keys())
                new_kwarg_map = {}
                for name, values_map in kwarg_map.items():
                    new_kwarg_map[name] = {
                        "index": parameter_names.index(name),
                        "values": values_map,
                    }
                _check_values(
                    version,
                    func.__qualname__,
                    args,
                    kwargs,
                    new_kwarg_map,
                    additional_msg,
                    stack_level,
                )
            return func(*args, **kwargs)

        return wrapper

    return decorator


def _deprecate_object(
    version: str,
    old_type: DeprecatedType,
    new_type: DeprecatedType,
    new_name: str,
    additional_msg: str,
    stack_level: int,
) -> Callable:
    """Decorator that prints deprecated message
    Args:
        version: Version to be used
        old_type: New type to be used
        new_type: New type to be used, if None, old_type is used instead.
        new_name: New name to be used
        additional_msg: any additional message
        stack_level: stack level

    Returns:
        The decorated method
    """

    def decorator(func):
        msg = (
            f"The {func.__name__} {old_type.value} is deprecated as of version {version} "
            "and will be removed no sooner than 3 months after the release"
        )
        if new_name:
            type_str = new_type.value if new_type is not None else old_type.value
            msg += f". Instead use the {new_name} {type_str}"
        if additional_msg:
            msg += f" {additional_msg}"
        msg += "."

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # warn only once
            if not wrapper._warned:
                warnings.warn(msg, DeprecationWarning, stacklevel=stack_level)
                wrapper._warned = True
            return func(*args, **kwargs)

        wrapper._warned = False
        return wrapper

    return decorator


def deprecate_method(
    version: str,
    new_type: Optional[DeprecatedType] = None,
    new_name: Optional[str] = None,
    additional_msg: Optional[str] = None,
    stack_level: int = 2,
) -> Callable:
    """Decorator that prints deprecated message for an instance method
    Args:
        version: Version to be used
        new_type: New type to be used
        new_name: New name to be used
        additional_msg: any additional message
        stack_level: stack level

    Returns:
        The decorated method
    """
    return _deprecate_object(
        version, DeprecatedType.METHOD, new_type, new_name, additional_msg, stack_level
    )


def deprecate_property(
    version: str,
    new_type: Optional[DeprecatedType] = None,
    new_name: Optional[str] = None,
    additional_msg: Optional[str] = None,
    stack_level: int = 2,
) -> Callable:
    """Decorator that prints deprecated message for a property

    *** This decorator must be placed below the property decorator ***

    Args:
        version: Version to be used
        new_type: New type to be used
        new_name: New name to be used
        additional_msg: any additional message
        stack_level: stack level

    Returns:
        The decorated property
    """
    return _deprecate_object(
        version, DeprecatedType.PROPERTY, new_type, new_name, additional_msg, stack_level
    )


def deprecate_function(
    version: str,
    new_type: Optional[DeprecatedType] = None,
    new_name: Optional[str] = None,
    additional_msg: Optional[str] = None,
    stack_level: int = 2,
) -> Callable:
    """Decorator that prints deprecated message for a function
    Args:
        version: Version to be used
        new_type: New type to be used
        new_name: New name to be used
        additional_msg: any additional message
        stack_level: stack level

    Returns:
        The decorated function
    """
    return _deprecate_object(
        version, DeprecatedType.FUNCTION, new_type, new_name, additional_msg, stack_level
    )
