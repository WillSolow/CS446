DMC_rs_test_angle_hist.py
Demonstrates that how we calculate the angle between the oxygen atoms in the water trimer is correct.


DMC_rs_test_arr_indexing.py
Demonstrates that the distinct atom pairing we use in the intermolecular PE function calculation is correct.


DMC_rs_test_fileload.py
Demonstrates how we load a file for the equilibrated walkers.


DMC_rs_test_inter_pe.py
Demonstrates that our intermolecular PE function is correct with respect to Prof Madison's PE function.
Note that our PE function is able to handle the edge case when two atoms are in the same position in 3 space
meaning the distance between them is 0.


DMC_rs_test_intra_pe.py
Demonstrates that our intramoleculear PE function is correct with respect to Prof Madison's PE function.


DMC_rs_test_propagations.py
Demonstrates that our propagating tiling method is correct. See that the coordinates with the larger
magnitudes are in the right positions.


DMC_rs_test_repl_delete_types.py
Demonstrates that replication and deletion is happening correctly by comparing our method to known output
using unit testing.


DMC_rs_test_replication.py
Another file to test that replication and deletion is done correctly. This is a manual test.


DMC_rs_test_total_pe.py
Would test our total PE vs Madison total PE but only works for a 1 walker system as Madison PE does not 
support multiple walkers.


DMC_rs_test_updated_Madison_pe.py
Extends Prof Madison's intermolecular PE function to multiple walkers and demonstrates its correctness with
respect to our PE function.