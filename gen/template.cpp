/* Copyright 2019-2020 SKA South Africa
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
 * This file is automatically generated. Do not edit.
 */

#ifndef _GNU_SOURCE
# define _GNU_SOURCE
#endif
#include <spead2/common_features.h>

#if {{ guard }}

#include <spead2/common_loader_{{ name }}.h>
#include <spead2/common_loader_utils.h>
#include <spead2/common_logging.h>
#include <mutex>
#include <exception>

namespace spead2
{

static std::once_flag init_once;
static std::exception_ptr init_result;

static void init();

{% for node in nodes %}
{{ node | rename(node.name + '_stub') | gen }}
{
{% for arg in (node | args) %}
    (void) {{ arg }};
{% endfor %}
    std::rethrow_exception(init_result);
}

{{ node | rename(node.name + '_first') | gen }}
{
    std::call_once(init_once, init);
    return {{ node.name }}({{ node | args | join(', ') }});
}

{% endfor %}

{% for node in nodes %}
{{ node | ptr | gen }} = {{ node.name }}_first;
{% endfor %}

static void reset_stubs()
{
{% for node in nodes %}
    {{ node.name }} = {{ node.name }}_stub;
{% endfor %}
}

static void init()
{
    try
    {
        dl_handle lib("{{ soname }}");
{% for node in nodes %}
        {{ node.name }} = reinterpret_cast<{{ node | ptr(False) | gen }}>(
            lib.sym("{{ node.name }}"));
{% endfor %}
        // Prevent the library being closed, so that the symbols stay valid
        lib.release();
    }
    catch (std::system_error &e)
    {
        init_result = std::current_exception();
        reset_stubs();
        log_warning("could not load {{ soname }}: %s", e.what());
    }
}

} // namespace spead2

{% if wrappers %}
/* Wrappers in the global namespace. This is needed because some functions
 * call others, and so we need to provide an implementation. We limit it
 * to functions where this is known to be an issue, because the header
 * doesn't always match the man page even though there is binary
 * compatibility (e.g. on LP64 systems, unsigned long and unsigned long long
 * are both 64-bit but are considered different types).
 */
{% for node in nodes if node.name in wrappers %}

{{ node | gen }}
{
    return spead2::{{ node.name }}({{ node | args | join(', ') }});
}
{% endfor %}
{% endif %}

#endif // SPEAD2_USE_IBV
