RUNNAME="$1"

echo Jobs remaining: `expr $(qstat | wc -l) - 2`

echo "Models completed: $(find /cluster/project2/gaml/trained/paper2/"$RUNNAME"_run -name '*.pt' | wc -l)"

echo "Empty log files: $(find /cluster/project2/gaml/trained/paper2/"$RUNNAME"_run -name '*.o' -size 0 | wc -l)"

echo "Files with possible errors: $(grep [A-Za-z]Error /cluster/project2/gaml/trained/paper2/"$RUNNAME"_run/"$RUNNAME"_logs/*.e -s | wc -l)"
