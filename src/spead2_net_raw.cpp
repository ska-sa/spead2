/* Copyright 2021 National Research Foundation (SARAO)
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

/* This file is written in C. It's in a .cpp file so that the build system
 * doesn't need to worry about setting the right compiler flags for both C
 * and C++.
 */

#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <sys/capability.h>
#include <sys/prctl.h>

static void check(int result, const char *name)
{
    if (result != 0)
    {
        perror(name);
        exit(1);
    }
}

int main(int argc, char **argv)
{
    int result;
    cap_t cap;
    const cap_value_t value = CAP_NET_RAW;

    if (argc < 2)
    {
        fprintf(stderr, "Usage: spead2_net_raw <program> [<args>...]\n");
        return 2;
    }

    cap = cap_get_proc();
    if (cap == NULL)
    {
        perror("cap_get_proc");
        return 1;
    }
    check(cap_set_flag(cap, CAP_INHERITABLE, 1, &value, CAP_SET), "cap_set_flag");
    check(cap_set_flag(cap, CAP_PERMITTED, 1, &value, CAP_SET), "cap_set_flag");
    result = cap_set_proc(cap);
    if (result != 0)
    {
        if (errno == EPERM)
        {
            fputs(
                "Permission denied. This probably means that you need to run\n"
                "\n"
                "    sudo setcap cap_net_raw+p /path/to/spead2_net_raw\n"
                "\n"
                "Please see the manual for more details, including security implications.\n",
                stderr);
        }
        else
            perror("cap_set_proc");
        exit(1);
    }
    cap_free(cap);

    /* Older versions of libcap don't support cap_set_ambient, so use prctl
     * directly.
     */
    check(prctl(PR_CAP_AMBIENT, PR_CAP_AMBIENT_RAISE, CAP_NET_RAW, 0, 0), "prctl");

    if (argc > 0)
    {
        argc--;
        argv++;
    }
    execvp(argv[0], argv);
    perror("execvp");
    return 1;
}
