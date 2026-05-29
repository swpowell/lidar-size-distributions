#!/bin/bash
#SBATCH --job-name=sgp-hypotheses
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --partition=ventana
#SBATCH --time=4-00:00:00
#SBATCH --mem=0
#SBATCH --output=SGP-hypothesis-pipeline-%j.txt

export LC_ALL=${LC_ALL:-}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
source /etc/profile
source ~/.bashrc
conda activate /home/scott.powell/anaconda3/envs/ventana

set -euo pipefail

REPO_DIR=${REPO_DIR:-/thumper/users/scott.powell/code/lidar-size-distributions}
DATA_ROOT=${DATA_ROOT:-/thumper/users/scott.powell/code-data/research-code/lidar/SGP}
CURATED_OUTPUT=${CURATED_OUTPUT:-${REPO_DIR}/SGP/curated_sgp_c1.parquet}
OUTPUT_DIR=${OUTPUT_DIR:-${REPO_DIR}/SGP/hypothesis_outputs}
CLOUDY_CSV=${CLOUDY_CSV:-${REPO_DIR}/SGP/bkn_sct_800_1500_no_lower_no_precip_no_fog.csv}
CLEAR_CSV=${CLEAR_CSV:-${REPO_DIR}/SGP/clear_skies.csv}
LOCATIONS=${LOCATIONS:-C1}
METHODS=${METHODS:-resampled}
OUTPUT_FORMAT=${OUTPUT_FORMAT:-parquet}
JOBS=${JOBS:-96}
LIDAR_JOBLIB_PREFER=${LIDAR_JOBLIB_PREFER:-processes}
RUN_GENERATION=${RUN_GENERATION:-1}
RUN_CURATION=${RUN_CURATION:-1}
RUN_ANALYSIS=${RUN_ANALYSIS:-1}
START_DATE=${START_DATE:-}
END_DATE=${END_DATE:-}
RESAMPLE_LEVELS=${RESAMPLE_LEVELS:-}
RESAMPLED_OUTPUT_TEMPLATE=${RESAMPLED_OUTPUT_TEMPLATE:-${DATA_ROOT}/C1_resample_level_{level}}

if [[ -n "${RESAMPLE_LEVELS}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    read -r -a resample_level_array <<< "${RESAMPLE_LEVELS}"
    if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#resample_level_array[@]} )); then
        echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} is outside RESAMPLE_LEVELS length ${#resample_level_array[@]}." >&2
        exit 1
    fi
    RESAMPLE_LEVELS=${resample_level_array[${SLURM_ARRAY_TASK_ID}]}
fi

cd "${REPO_DIR}"

echo "[$(date --iso-8601=seconds)] Starting SGP hypothesis pipeline"
echo "Repo: ${REPO_DIR}"
echo "Data root: ${DATA_ROOT}"
echo "Curated output: ${CURATED_OUTPUT}"
echo "Hypothesis output dir: ${OUTPUT_DIR}"
echo "Jobs: ${JOBS}"
echo "Methods: ${METHODS}"
echo "Output format: ${OUTPUT_FORMAT}"
if [[ -n "${RESAMPLE_LEVELS}" ]]; then
    echo "Resample levels: ${RESAMPLE_LEVELS}"
    echo "Resampled output template: ${RESAMPLED_OUTPUT_TEMPLATE}"
fi
echo "Joblib prefer: ${LIDAR_JOBLIB_PREFER}"
echo "Native library threads: OMP=${OMP_NUM_THREADS}, OPENBLAS=${OPENBLAS_NUM_THREADS}, MKL=${MKL_NUM_THREADS}, NUMEXPR=${NUMEXPR_NUM_THREADS}"
export LIDAR_JOBLIB_PREFER

if [[ "${RUN_GENERATION}" == "1" ]]; then
    if [[ -n "${RESAMPLE_LEVELS}" ]]; then
        for resample_level in ${RESAMPLE_LEVELS}; do
            resampled_output_dir=${RESAMPLED_OUTPUT_TEMPLATE/\{level\}/${resample_level}}
            generation_args=(
                --jobs "${JOBS}"
                --methods ${METHODS}
                --output-format "${OUTPUT_FORMAT}"
                --resample-level "${resample_level}"
                --regular-output-dir "${resampled_output_dir}"
                --resampled-output-dir "${resampled_output_dir}"
            )
            if [[ -n "${START_DATE}" ]]; then
                generation_args+=(--start-date "${START_DATE}")
            fi
            if [[ -n "${END_DATE}" ]]; then
                generation_args+=(--end-date "${END_DATE}")
            fi
            echo "[$(date --iso-8601=seconds)] Generating daily event tables for resample level ${resample_level}"
            printf 'Command: python -u SGP/generate_event_tables.py'
            printf ' %q' "${generation_args[@]}"
            printf '\n'
            python -u SGP/generate_event_tables.py "${generation_args[@]}"
        done
    else
        generation_args=(--jobs "${JOBS}" --methods ${METHODS} --output-format "${OUTPUT_FORMAT}")
        if [[ -n "${START_DATE}" ]]; then
            generation_args+=(--start-date "${START_DATE}")
        fi
        if [[ -n "${END_DATE}" ]]; then
            generation_args+=(--end-date "${END_DATE}")
        fi
        echo "[$(date --iso-8601=seconds)] Generating daily event tables"
        printf 'Command: python -u SGP/generate_event_tables.py'
        printf ' %q' "${generation_args[@]}"
        printf '\n'
        python -u SGP/generate_event_tables.py "${generation_args[@]}"
    fi
else
    echo "[$(date --iso-8601=seconds)] Skipping event-table generation"
fi

if [[ "${RUN_CURATION}" == "1" ]]; then
    echo "[$(date --iso-8601=seconds)] Building curated dataset"
    printf 'Command: python -u SGP/build_curated_dataset.py --root-dir %q --locations %s --methods %s --input-format %q --cloudy-csv %q --clear-csv %q --output %q\n' \
        "${DATA_ROOT}" "${LOCATIONS}" "${METHODS}" "${OUTPUT_FORMAT}" "${CLOUDY_CSV}" "${CLEAR_CSV}" "${CURATED_OUTPUT}"
    python -u SGP/build_curated_dataset.py \
        --root-dir "${DATA_ROOT}" \
        --locations ${LOCATIONS} \
        --methods ${METHODS} \
        --input-format "${OUTPUT_FORMAT}" \
        --cloudy-csv "${CLOUDY_CSV}" \
        --clear-csv "${CLEAR_CSV}" \
        --output "${CURATED_OUTPUT}"
else
    echo "[$(date --iso-8601=seconds)] Skipping curated dataset build"
fi

if [[ "${RUN_ANALYSIS}" == "1" ]]; then
    echo "[$(date --iso-8601=seconds)] Running H1/H2/H3 analyses"
    printf 'Command: python -u SGP/run_all_hypothesis_analyses.py --input %q --output-dir %q\n' \
        "${CURATED_OUTPUT}" "${OUTPUT_DIR}"
    python -u SGP/run_all_hypothesis_analyses.py \
        --input "${CURATED_OUTPUT}" \
        --output-dir "${OUTPUT_DIR}"
else
    echo "[$(date --iso-8601=seconds)] Skipping hypothesis analyses"
fi

echo "[$(date --iso-8601=seconds)] Finished SGP hypothesis pipeline"
