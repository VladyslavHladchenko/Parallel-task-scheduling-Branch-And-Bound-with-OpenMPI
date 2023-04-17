### Optimal task scheduling. 
For a number of tasks, where each is described by its start time, task duration and task deadline, the branch and bound algorithm finds an optimal schedule or indicates if such a schedule is impossible.

#### Input format
```
n
duration release deadline
```
duration release deadline are integers.

n - total number of tasks, line i describes the i-th task

#### Output format
```
s_0
s_1
...
s_{n-1}
```
s_i - start time of task t_i.

Output contains single line -1 if the problem instance is infeasible.

### Build
```
cd cmake
cmake .
make
```

### Run

```
mpiexec -np NUM_PROCESSES ./bratley INSTANCE_PATH OUTPUT_PATH
```