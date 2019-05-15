/* Copyright 2019 SKA South Africa
 *
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/**
 * @file
 *
 * Unit tests for common_logging.
 */

#include <boost/test/unit_test.hpp>
#include <spead2/common_logging.h>
#include <vector>
#include <string>
#include <utility>
#include <system_error>
#include <iostream>

namespace spead2
{
namespace unittest
{

struct capture_logging
{
    std::function<void(log_level, const std::string &)> orig_log_function;
    std::vector<log_level> levels;
    std::vector<std::string> messages;

    capture_logging()
        : orig_log_function(set_log_function(
            [this] (log_level level, const std::string &msg)
            {
                levels.push_back(level);
                messages.push_back(msg);
            }))
    {
    }

    ~capture_logging()
    {
        set_log_function(orig_log_function);
    }
};

struct capture_stderr
{
    std::ostringstream out;
    std::streambuf *old_buf;

    capture_stderr()
    {
        old_buf = std::cerr.rdbuf(out.rdbuf());
    }

    ~capture_stderr()
    {
        std::cerr.rdbuf(old_buf);
    }
};

BOOST_AUTO_TEST_SUITE(common)
BOOST_FIXTURE_TEST_SUITE(logging, capture_logging)

#define CHECK_MESSAGES(message, level) do {                                                \
        std::vector<std::string> expected_messages{(message)};                             \
        BOOST_CHECK_EQUAL_COLLECTIONS(messages.begin(), messages.end(),                    \
                                      expected_messages.begin(), expected_messages.end()); \
        std::vector<log_level> expected_levels{(level)};                                   \
        BOOST_CHECK_EQUAL_COLLECTIONS(levels.begin(), levels.end(),                        \
                                      expected_levels.begin(), expected_levels.end());     \
    } while (false)

BOOST_AUTO_TEST_CASE(log_info)
{
    spead2::log_info("Hello %1%", 3);
    CHECK_MESSAGES("Hello 3", log_level::info);
}

BOOST_AUTO_TEST_CASE(log_errno_explicit)
{
    errno = 0;
    log_errno("Test: %1% %2%", EBADF);
    std::ostringstream expected;
    expected << "Test: " << EBADF << " Bad file descriptor";
    CHECK_MESSAGES(expected.str(), log_level::warning);
}

BOOST_AUTO_TEST_CASE(log_errno_implicit)
{
    errno = EBADF;
    spead2::log_errno("Test: %1% %2%");
    std::ostringstream expected;
    expected << "Test: " << EBADF << " Bad file descriptor";
    CHECK_MESSAGES(expected.str(), log_level::warning);
}

static boost::test_tools::predicate_result ebadf_exception(const std::system_error &error)
{
    boost::test_tools::predicate_result result(false);
    if (error.code().value() != EBADF)
    {
        result.message() << "Expected error code " << EBADF << ", got " << error.code().value();
        return result;
    }
    if (error.code().category() != std::system_category())
    {
        result.message() << "Incorrect error category";
        return result;
    }
    std::string what = error.what();
    if (what.find("blah") == std::string::npos)
    {
        result.message() << "Did not find 'blah' in '" << what << "'";
        return result;
    }
    return true;
}

static boost::test_tools::predicate_result no_error_exception(const std::system_error &error)
{
    boost::test_tools::predicate_result result(false);
    if (error.code().default_error_condition() != std::errc::invalid_argument)
    {
        result.message() << "Expected error condition invalid_argument, got "
            << error.code().value();
        return result;
    }
    std::string what = error.what();
    if (what.find("blah") == std::string::npos)
    {
        result.message() << "Did not find 'blah' in '" << what << "'";
        return result;
    }
    if (what.find("(unknown error)") == std::string::npos)
    {
        result.message() << "Did not find '(unknown error)' in '" << what << "'";
        return result;
    }
    return true;
}

BOOST_AUTO_TEST_CASE(throw_errno_explicit)
{
    errno = 0;
    BOOST_CHECK_EXCEPTION(throw_errno("blah", EBADF), std::system_error, ebadf_exception);
}

BOOST_AUTO_TEST_CASE(throw_errno_implicit)
{
    errno = EBADF;
    BOOST_CHECK_EXCEPTION(throw_errno("blah"), std::system_error, ebadf_exception);
}

BOOST_AUTO_TEST_CASE(throw_errno_zero)
{
    BOOST_CHECK_EXCEPTION(throw_errno("blah", 0), std::system_error, no_error_exception);
}

BOOST_FIXTURE_TEST_CASE(default_logger, capture_stderr)
{
    spead2::log_info("A test message");
    BOOST_CHECK_EQUAL(out.str(), "spead2: info: A test message\n");
}

BOOST_AUTO_TEST_SUITE_END()  // logging
BOOST_AUTO_TEST_SUITE_END()  // common

}} // namespace spead2::unittest
