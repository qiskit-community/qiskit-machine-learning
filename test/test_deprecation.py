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

"""
Test The deprecation methods
"""

import unittest
import sys
import inspect
import warnings
from typing import Tuple, Optional
from test import QiskitMachineLearningTestCase
from ddt import data, ddt
from qiskit_machine_learning.deprecation import (
    DeprecatedEnum,
    DeprecatedEnumMeta,
    DeprecatedType,
    warn_deprecated,
    warn_deprecated_same_type_name,
    deprecate_function,
    deprecate_property,
    deprecate_method,
    deprecate_arguments,
    deprecate_values,
)

# pylint: disable=bad-docstring-quotes


class EnumTest(DeprecatedEnum, metaclass=DeprecatedEnumMeta):
    """Enumeration Test"""

    ONE = "one"
    TWO = "two"

    def deprecate(self):
        """show deprecate message"""
        warn_deprecated(
            "0.2.0", DeprecatedType.ENUM, self.__class__.__name__, new_name="NewEnum", stack_level=3
        )


@deprecate_function("0.1.1", DeprecatedType.CLASS, "some_class", "and more information")
def func1(arg1: int) -> int:
    """function 1"""
    return arg1


@deprecate_function("0.2.0", new_name="some_function2")
def func2(arg2: int) -> int:
    """function 2"""
    return arg2


@deprecate_arguments("0.1.2", {"old_arg": "new_arg"})
def func3(new_arg: Optional[int] = None, old_arg: Optional[int] = None) -> Tuple[int, int]:
    """function 3"""
    return new_arg, old_arg


class DeprecatedClass1:
    """Deprecated Test class 1"""

    def __init__(self):
        warn_deprecated(
            "0.3.0", DeprecatedType.CLASS, "DeprecatedClass1", DeprecatedType.CLASS, "NewClass"
        )
        self.value = 10


class DeprecatedClass2:
    """Deprecated Test class 2"""

    def __init__(self):
        warn_deprecated_same_type_name(
            "0.3.0", DeprecatedType.CLASS, "DeprecatedClass2", "from package test2"
        )
        self.value = 10


class DeprecatedClass3:
    """Deprecated Test class 3"""

    @deprecate_values("0.2.0", {"loss": {"l1": "absolute_error", "l2": "squared_error"}})
    def __init__(self, arg1: int, loss: str = "squared_error") -> None:
        self.value = arg1
        self.loss = loss

    @deprecate_values("0.2.0", {"loss": {"l1": "absolute_error", "l2": "squared_error"}})
    def method1(self, arg1: int, loss: str = "squared_error") -> None:
        """method 1"""
        self.value = arg1
        self.loss = loss


class DeprecatedClass4:
    """Deprecated Test class 4"""

    def __init__(
        self,
    ) -> None:
        self.status = 2

    @deprecate_values("0.2.0", {"status": {3: 2, 4: 5}})
    def method1(self, status: int = 2) -> None:
        """method 1"""
        self.status = status


class TestClass:
    """Test class with deprecation"""

    def __init__(self):
        self._value = 20

    # Bug in mypy, if property decorator is used with another one
    # https://github.com/python/mypy/issues/1362

    @property  # type: ignore
    @deprecate_property("0.1.0")
    def property1(self) -> int:
        """property1 get"""
        return self._value

    @property1.setter  # type: ignore
    @deprecate_property(
        "0.1.0", new_name="new_property", additional_msg="and some additional information"
    )
    def property1(self, value: int):
        """property 1 set"""
        self._value = value

    @deprecate_method("0.1.0", new_name="some_method1", additional_msg="and additional information")
    def method1(self, arg: int) -> int:
        """method 1"""
        return arg

    @deprecate_method("0.2.0", new_name="some_method2")
    def method2(self, arg: int) -> int:
        """method 2"""
        return arg

    @deprecate_arguments("0.1.2", {"old_arg": "new_arg"})
    def method3(
        self, new_arg: Optional[int] = None, old_arg: Optional[int] = None
    ) -> Tuple[int, int]:
        """method3"""
        return new_arg, old_arg


@ddt
class TestDeprecation(QiskitMachineLearningTestCase):
    """Test deprecation methods"""

    def setUp(self) -> None:
        super().setUp()
        self._source = inspect.getsource(sys.modules[self.__module__]).splitlines()

    def _get_line_from_str(self, text: str) -> int:
        for idx, line in enumerate(self._source):
            if text in line:
                return idx + 1
        return -1

    def test_enum_deprecation(self):
        """test enumeration deprecation"""

        msg_ref = (
            "The EnumTest enum is deprecated as of version 0.2.0 "
            "and will be removed no sooner than 3 months after the release. "
            "Instead use the NewEnum enum."
        )

        # emit deprecation the first time it is used
        with warnings.catch_warnings(record=True) as c_m:
            warnings.simplefilter("always")
            _ = EnumTest.ONE
            msg = str(c_m[0].message)
            self.assertEqual(msg, msg_ref)
            self.assertTrue("test_deprecation.py" in c_m[0].filename, c_m[0].filename)
            self.assertEqual(self._get_line_from_str("EnumTest.ONE"), c_m[0].lineno)

        # trying again should not emit deprecation
        with warnings.catch_warnings(record=True) as c_m:
            warnings.simplefilter("always")
            _ = EnumTest.TWO
            self.assertListEqual(c_m, [])

    @data(
        (
            "func1",
            "The func1 function is deprecated as of version 0.1.1 "
            "and will be removed no sooner than 3 months after the release. "
            "Instead use the some_class class and more information.",
        ),
        (
            "func2",
            "The func2 function is deprecated as of version 0.2.0 "
            "and will be removed no sooner than 3 months after the release. "
            "Instead use the some_function2 function.",
        ),
    )
    def test_function_deprecation(self, config):
        """test function deprecation"""

        function_name, msg_ref = config

        # emit deprecation the first time it is used
        with warnings.catch_warnings(record=True) as c_m:
            warnings.simplefilter("always")
            self.assertEqual(2, globals()[function_name](2))
            msg = str(c_m[0].message)
            self.assertEqual(msg, msg_ref)
            self.assertTrue("test_deprecation.py" in c_m[0].filename, c_m[0].filename)
            self.assertEqual(self._get_line_from_str("globals()[function_name](2)"), c_m[0].lineno)

        # trying again should not emit deprecation
        with warnings.catch_warnings(record=True) as c_m:
            warnings.simplefilter("always")
            globals()[function_name](None)
            self.assertListEqual(c_m, [])

    def test_class_deprecation1(self):
        """test class deprecation 1"""

        msg_ref = (
            "The DeprecatedClass1 class is deprecated as of version 0.3.0 "
            "and will be removed no sooner than 3 months after the release. "
            "Instead use the NewClass class."
        )

        # emit deprecation the first time it is used
        with warnings.catch_warnings(record=True) as c_m:
            warnings.simplefilter("always")
            obj = DeprecatedClass1()
            self.assertEqual(obj.value, 10)
            msg = str(c_m[0].message)
            self.assertEqual(msg, msg_ref)
            self.assertTrue("test_deprecation.py" in c_m[0].filename, c_m[0].filename)
            self.assertEqual(self._get_line_from_str("DeprecatedClass1()"), c_m[0].lineno)

        # trying again should not emit deprecation
        with warnings.catch_warnings(record=True) as c_m:
            warnings.simplefilter("always")
            obj = DeprecatedClass1()
            self.assertEqual(obj.value, 10)
            self.assertListEqual(c_m, [])

    def test_class_deprecation2(self):
        """test class deprecation 2"""

        msg_ref = (
            "The DeprecatedClass2 class is deprecated as of version 0.3.0 "
            "and will be removed no sooner than 3 months after the release. "
            "Instead use the DeprecatedClass2 class from package test2."
        )

        # emit deprecation the first time it is used
        with warnings.catch_warnings(record=True) as c_m:
            warnings.simplefilter("always")
            obj = DeprecatedClass2()
            self.assertEqual(obj.value, 10)
            msg = str(c_m[0].message)
            self.assertEqual(msg, msg_ref)
            self.assertTrue("test_deprecation.py" in c_m[0].filename, c_m[0].filename)
            self.assertEqual(self._get_line_from_str("DeprecatedClass2()"), c_m[0].lineno)

        # trying again should not emit deprecation
        with warnings.catch_warnings(record=True) as c_m:
            warnings.simplefilter("always")
            obj = DeprecatedClass2()
            self.assertEqual(obj.value, 10)
            self.assertListEqual(c_m, [])

    @data(
        (
            "l1",
            'The loss argument value "l1" is deprecated as of version 0.2.0 '
            "and will be removed no sooner than 3 months after the release. "
            'Instead use the "absolute_error" value.',
        ),
        (
            "l2",
            'The loss argument value "l2" is deprecated as of version 0.2.0 '
            "and will be removed no sooner than 3 months after the release. "
            'Instead use the "squared_error" value.',
        ),
    )
    def test_string_values_deprecation(self, config):
        """test string values deprecation"""

        loss, msg_ref = config

        # emit deprecation the first time it is used
        with warnings.catch_warnings(record=True) as c_m:
            warnings.simplefilter("always")
            obj = DeprecatedClass3(10, loss=loss)
            self.assertEqual(obj.value, 10)
            msg = str(c_m[0].message)
            self.assertEqual(msg, msg_ref)
            self.assertTrue("test_deprecation.py" in c_m[0].filename, c_m[0].filename)
            self.assertEqual(
                self._get_line_from_str("DeprecatedClass3(10, loss=loss)"), c_m[0].lineno
            )

        # trying again should not emit deprecation
        with warnings.catch_warnings(record=True) as c_m:
            warnings.simplefilter("always")
            obj = DeprecatedClass3(10, loss=loss)
            self.assertEqual(obj.value, 10)
            self.assertListEqual(c_m, [])

        # emit deprecation the first time it is used
        with warnings.catch_warnings(record=True) as c_m:
            warnings.simplefilter("always")
            obj = DeprecatedClass3(5, loss=loss)
            obj.method1(10, loss)
            self.assertEqual(obj.value, 10)
            msg = str(c_m[0].message)
            self.assertEqual(msg, msg_ref)
            self.assertTrue("test_deprecation.py" in c_m[0].filename, c_m[0].filename)
            self.assertEqual(self._get_line_from_str("obj.method1(10, loss)"), c_m[0].lineno)

        # trying again should not emit deprecation
        with warnings.catch_warnings(record=True) as c_m:
            warnings.simplefilter("always")
            obj = DeprecatedClass3(5, loss=loss)
            obj.method1(10, loss)
            self.assertEqual(obj.value, 10)
            self.assertListEqual(c_m, [])

    @data(
        (
            3,
            'The status argument value "3" is deprecated as of version 0.2.0 '
            "and will be removed no sooner than 3 months after the release. "
            'Instead use the "2" value.',
        ),
        (
            4,
            'The status argument value "4" is deprecated as of version 0.2.0 '
            "and will be removed no sooner than 3 months after the release. "
            'Instead use the "5" value.',
        ),
    )
    def test_int_values_deprecation(self, config):
        """test int values deprecation"""

        status, msg_ref = config
        obj = DeprecatedClass4()

        # emit deprecation the first time it is used
        with warnings.catch_warnings(record=True) as c_m:
            warnings.simplefilter("always")
            obj.method1(status)
            msg = str(c_m[0].message)
            self.assertEqual(msg, msg_ref)
            self.assertTrue("test_deprecation.py" in c_m[0].filename, c_m[0].filename)
            self.assertEqual(self._get_line_from_str("obj.method1(status)"), c_m[0].lineno)

        # trying again should not emit deprecation
        with warnings.catch_warnings(record=True) as c_m:
            warnings.simplefilter("always")
            obj.method1(status)
            self.assertEqual(obj.status, status)
            self.assertListEqual(c_m, [])

    @data(
        (
            "method1",
            "The method1 method is deprecated as of version 0.1.0 "
            "and will be removed no sooner than 3 months after the release. "
            "Instead use the some_method1 method and additional information.",
        ),
        (
            "method2",
            "The method2 method is deprecated as of version 0.2.0 "
            "and will be removed no sooner than 3 months after the release. "
            "Instead use the some_method2 method.",
        ),
    )
    def test_method_deprecation(self, config):
        """test method deprecation"""

        method_name, msg_ref = config
        method = getattr(TestClass(), method_name)

        # emit deprecation the first time it is used
        with warnings.catch_warnings(record=True) as c_m:
            warnings.simplefilter("always")
            self.assertEqual(3, method(3))
            msg = str(c_m[0].message)
            self.assertEqual(msg, msg_ref)
            self.assertTrue("test_deprecation.py" in c_m[0].filename, c_m[0].filename)
            self.assertEqual(self._get_line_from_str("method(3)"), c_m[0].lineno)

        # trying again should not emit deprecation
        with warnings.catch_warnings(record=True) as c_m:
            warnings.simplefilter("always")
            self.assertEqual(4, method(4))
            self.assertListEqual(c_m, [])

    def test_function_arguments_deprecation(self):
        """test function arguments deprecation"""

        msg_ref = (
            "func3: the old_arg argument is deprecated as of version 0.1.2 "
            "and will be removed no sooner than 3 months after the release. "
            "Instead use the new_arg argument."
        )
        # both arguments at the same time should raise exception
        with self.assertRaises(TypeError):
            func3(new_arg="2222", old_arg="hello")

        with warnings.catch_warnings(record=True) as c_m:
            warnings.simplefilter("always")
            self.assertEqual(("hello", None), func3(old_arg="hello"))
            msg = str(c_m[0].message)
            self.assertEqual(msg, msg_ref)
            self.assertTrue("test_deprecation.py" in c_m[0].filename, c_m[0].filename)
            self.assertEqual(self._get_line_from_str('func3(old_arg="hello")'), c_m[0].lineno)

        # trying again should not emit deprecation
        with warnings.catch_warnings(record=True) as c_m:
            warnings.simplefilter("always")
            self.assertEqual(("hello", None), func3(old_arg="hello"))
            self.assertListEqual(c_m, [])

    def test_method_arguments_deprecation(self):
        """test method arguments deprecation"""

        obj = TestClass()

        msg_ref = (
            "method3: the old_arg argument is deprecated as of version 0.1.2 "
            "and will be removed no sooner than 3 months after the release. "
            "Instead use the new_arg argument."
        )
        # both arguments at the same time should raise exception
        with self.assertRaises(TypeError):
            obj.method3(new_arg="2222", old_arg="hello")

        with warnings.catch_warnings(record=True) as c_m:
            warnings.simplefilter("always")
            self.assertEqual(("hello", None), obj.method3(old_arg="hello"))
            msg = str(c_m[0].message)
            self.assertEqual(msg, msg_ref)
            self.assertTrue("test_deprecation.py" in c_m[0].filename, c_m[0].filename)
            self.assertEqual(self._get_line_from_str('obj.method3(old_arg="hello")'), c_m[0].lineno)

        # trying again should not emit deprecation
        with warnings.catch_warnings(record=True) as c_m:
            warnings.simplefilter("always")
            self.assertEqual(("hello", None), obj.method3(old_arg="hello"))
            self.assertListEqual(c_m, [])

    def test_property_deprecation(self):
        """test property deprecation"""

        obj = TestClass()

        msg_ref = (
            "The property1 property is deprecated as of version 0.1.0 "
            "and will be removed no sooner than 3 months after the release."
        )
        # property get
        # emit deprecation the first time it is used
        with warnings.catch_warnings(record=True) as c_m:
            warnings.simplefilter("always")
            self.assertEqual(20, obj.property1)
            msg = str(c_m[0].message)
            self.assertEqual(msg, msg_ref)
            self.assertTrue("test_deprecation.py" in c_m[0].filename, c_m[0].filename)
            self.assertEqual(self._get_line_from_str("obj.property1"), c_m[0].lineno)

        # trying again should not emit deprecation
        with warnings.catch_warnings(record=True) as c_m:
            warnings.simplefilter("always")
            self.assertEqual(20, obj.property1)
            self.assertListEqual(c_m, [])

        msg_ref = (
            "The property1 property is deprecated as of version 0.1.0 "
            "and will be removed no sooner than 3 months after the release. "
            "Instead use the new_property property and some additional information."
        )
        # property set
        # emit deprecation the first time it is used
        with warnings.catch_warnings(record=True) as c_m:
            warnings.simplefilter("always")
            obj.property1 = 0
            self.assertEqual(0, obj.property1)
            msg = str(c_m[0].message)
            self.assertEqual(msg, msg_ref)
            self.assertTrue("test_deprecation.py" in c_m[0].filename, c_m[0].filename)
            self.assertEqual(self._get_line_from_str("obj.property1 = 0"), c_m[0].lineno)

        # trying again should not emit deprecation
        with warnings.catch_warnings(record=True) as c_m:
            warnings.simplefilter("always")
            obj.property1 = 0
            self.assertEqual(0, obj.property1)
            self.assertListEqual(c_m, [])


if __name__ == "__main__":
    unittest.main()
