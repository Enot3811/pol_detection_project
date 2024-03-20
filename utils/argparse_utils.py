"""Module contains useful utils for argparse."""


import argparse
from typing import Sequence, Any, Union, Optional, Callable
from functools import partial


def positive_type(
    value: str, cast_f: Callable[[str], Any], include_zero: bool = False
) -> Any:
    """
    Wrap type for `argparse`.

    This function is a type for `argparse` package argument.
    This type has additional limitation on range of values - only positive.

    Parameters
    ----------
    value : str
        Value that will be processed by type-cast.
    cast_f : Callable[[str], Any]
        Function that makes base-cast. Typical values are: `int`, `float`.
        Casted type should support simple comparison.
    include_zero : bool, optional
        Bool flag that controls zero include in OK-range.

    Returns
    -------
    Any
        Argument casted with `cast_f`.

    Raises
    ------
    ValueError:
        This exception will be raised if casted value is not positive.
    """
    new_value = cast_f(value)
    if include_zero:
        check_f = lambda x: x >= 0
    else:
        check_f = lambda x: x > 0
    if not check_f(new_value):
        raise IOError(f'Argument that must be positive has negative '
                      f'value: "{new_value}".')
    return new_value


natural_int = partial(positive_type, cast_f=int)
natural_float = partial(positive_type, cast_f=float)
non_negative_int = partial(positive_type, cast_f=int, include_zero=True)
non_negative_float = partial(positive_type, cast_f=float, include_zero=True)


def unit_interval(value: str) -> float:
    """
    Wrap type for `argparse`. This type defines float in `[0..1]`.

    This function is a type for `argparse` package argument.

    Parameters
    ----------
    value : str
        Value that will be processed by type-cast.

    Returns
    -------
    float
        Float value in unit interval.

    Raises
    ------
    ValueError
        This exception will be raised if casted value is not positive.
    """
    new_value = float(value)
    if new_value < 0 or new_value > 1:
        raise IOError(f'Argument that must be in [0..1] has wrong '
                      f'value: "{new_value}".')
    return new_value


def required_length(min_num: int, max_num: int) -> argparse.Action:
    """Action to make boundaries for nargs "+" in `argparse`.

    Parameters
    ----------
    num_min : int
        Number of minimum required arguments. Including this bound.
    num_max : int
        Number of maximum required arguments. Including this bound.
    
    Returns
    -------
    argparse.Action
        Action that make boundaries for nargs "+" in `argparse`.

    Raises
    ------
    argparse.ArgumentTypeError
        Raise if number of passed arguments is out of interval
        `[min_num, max_num]`.
    """
    class RequiredLength(argparse.Action):
        def __call__(
            self, parser: argparse.ArgumentParser,
            namespace: argparse.Namespace,
            values: Optional[Union[str | Sequence[Any]]],
            option_string: Optional[str] = None
        ):
            if not min_num <= len(values) <= max_num:
                raise argparse.ArgumentTypeError(
                    f'Number of arguments "{self.dest}" required to be '
                    f'between {min_num} and {max_num}.')
            setattr(namespace, self.dest, values)
    return RequiredLength
