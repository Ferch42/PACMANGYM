from LTL import prog


T = ("UNTIL", "TRUE", ("AND", "TOUCHED", "FLAG"))
META = ("UNTIL", "TRUE", ("AND", T, ("NEXT", T) ))


print(META)
print(prog({'TOUCHED', "FLAG"}, META))
print(prog({'TOUCHED', "FLAG"},prog({'TOUCHED', "FLAG"}, META)))