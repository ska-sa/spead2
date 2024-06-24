# Copyright 2023-2024 National Research Foundation (SARAO)
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import inspect

import pytest


@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item, nextitem):
    # Workaround for https://github.com/pytest-dev/pytest/issues/11374
    if inspect.ismethod(item.obj):
        item.obj.__self__.__dict__.clear()


@pytest.hookimpl
def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "transport(filter): add a filter on the transport classes")
