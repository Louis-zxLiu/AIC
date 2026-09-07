#!/bin/bash
set -euo pipefail
cd /public/home/jqfan/AIC
SBATCH=/opt/gridview/slurm/bin/sbatch
IR_JOB=$($SBATCH --parsable automation/sugon_teacher_ir_128.slurm)
DEPTH_JOB=$($SBATCH --parsable --dependency=afterok:${IR_JOB} automation/sugon_teacher_depth_128.slurm)
STUDENT_JOB=$($SBATCH --parsable --dependency=afterok:${DEPTH_JOB} automation/sugon_student_infer_128.slurm)
echo "IR_JOB=${IR_JOB}"
echo "DEPTH_JOB=${DEPTH_JOB}"
echo "STUDENT_JOB=${STUDENT_JOB}"
