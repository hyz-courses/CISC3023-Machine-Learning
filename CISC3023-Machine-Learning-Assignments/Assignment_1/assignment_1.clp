(deffacts MAIN::known "Known facts"
   (A)
   (B)
   (C))

(defrule MAIN::rule1 "1"
   (A)
   (B)
   =>
   (assert (E)))

(defrule MAIN::rule2 "2"
   (B)
   (C)
   =>
   (assert (F)))

(defrule MAIN::rule3 "3"
   (E)
   (F)
   (C)
   =>
   (assert (G)))

