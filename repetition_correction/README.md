# How to do the repetition correction

Once you have a sequence (as a list, such as [0,1,2,3,..]), do correct_nodes(sequence). It will return the corrected sequence via the method of repetition correction

To integrate it into the code every few iterations, we do a sanity check using the sanity_check on the sequences every 50 (or other appropriate #) iterations. 
If the sanity check fails, it means we have repetitions and we correct them using function above. 

The files also contain additional utils functions to convert the binary string to sequences and sequences to binaries, etc. 

