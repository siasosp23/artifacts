#!/bin/bash
# clear adaptdl checkpoints
rm -rf /hot/suhasj-pollux-pvc-1d0de4e8-040c-4515-a06e-7d5629e06c21/pollux/checkpoint/*/.* 2>/dev/null
rm -rf /hot/suhasj-pollux-pvc-1d0de4e8-040c-4515-a06e-7d5629e06c21/pollux/checkpoint/*/*  2>/dev/null
# clear adaptdl job-initiated checkpoints
rm -rf /hot/suhasj-pollux-pvc-1d0de4e8-040c-4515-a06e-7d5629e06c21/pollux/checkpoint-job/*/.*  2>/dev/null
rm -rf /hot/suhasj-pollux-pvc-1d0de4e8-040c-4515-a06e-7d5629e06c21/pollux/checkpoint-job/*/*  2>/dev/null
# clear tensorboard
rm -rf /hot/suhasj-pollux-pvc-1d0de4e8-040c-4515-a06e-7d5629e06c21/pollux/tensorboard/*/.*  2>/dev/null
rm -rf /hot/suhasj-pollux-pvc-1d0de4e8-040c-4515-a06e-7d5629e06c21/pollux/tensorboard/*/*  2>/dev/null
