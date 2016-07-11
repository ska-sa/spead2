#serial 1

# SPEAD2_ARG_WITH(ARG, HELP, VAR, TEST)
# ----------------------------------------
# 
AC_DEFUN([SPEAD2_ARG_WITH],
         [AS_VAR_PUSHDEF(var, [$3])
          AS_VAR_PUSHDEF(wvar, [with_$1])
          AS_VAR_SET([var], [0])
          AC_ARG_WITH([$1], [$2], [], [AS_VAR_SET([wvar], [check])])
          AS_VAR_IF([wvar], [no], [],
              [$4
               AS_VAR_IF([wvar], [check], [],
                   [AS_VAR_IF([var], [0],
                        [AC_MSG_ERROR([--with-$1 was specified but not found])])])
              ])
          AC_SUBST($3)
          AS_VAR_POPDEF([var])
          AS_VAR_POPDEF([wvar])
        ])
