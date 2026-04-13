#!/bin/bash
# run_eval_sweep.sh
# Runs eval for all perturbed variants across all test sets for both baseline and FiLM.
# Skips the base robot (set="base" / name="g1_12dof").
# All output written to a single log file with clear section separators.

cd /home/sviswasam/dr/unitree_rl_gym

BASELINE_CKPT="/home/sviswasam/dr/unitree_rl_gym/output_baseline_results2/Apr11_18-52-43/model_400.pt"
FILM_CKPT="/home/sviswasam/dr/unitree_rl_gym/output_film_results2/Apr11_18-53-28/model_400.pt"
XML_PATH="/home/sviswasam/dr/ModuMorph/modular/unitree_g1_actual/xml/g1_12dof_stripped.xml"
OOD_ROOT="/home/sviswasam/dr/unitree_rl_gym/resources/robots/g1_ood_test_sets2"
LOG_FILE="/home/sviswasam/dr/unitree_rl_gym/logs/eval_sweep_$(date +%Y%m%d_%H%M%S).log"

TEST_SETS=(
    "armature_perturbed"
    "effort_perturbed"
    "mass_perturbed"
    "joint_range_perturbed"
    "all_perturbed"
    "damping_perturbed"
)

log() {
    echo "$@" | tee -a "$LOG_FILE"
}

run_variant() {
    local model_label="$1"   # "BASELINE" or "FILM"
    local ckpt="$2"
    local baseline_flag="$3" # "--baseline" or ""
    local test_set="$4"
    local variant_name="$5"
    local metadata="$6"

    log ""
    log "────────────────────────────────────────────────────────"
    log "  Model      : $model_label"
    log "  Test set   : $test_set"
    log "  Variant    : $variant_name"
    log "────────────────────────────────────────────────────────"

    python deploy/deploy_mujoco/record_traj_isaac.py \
        --checkpoint "$ckpt" \
        --xml_path "$XML_PATH" \
        --variants_metadata "$metadata" \
        --variant_name "$variant_name" \
        --num_eval_rollouts 10 \
        --cmd_vx 0.5 \
        --out /tmp/traj_${model_label}_${test_set}_${variant_name}.pkl \
        $baseline_flag 2>&1 | tee -a "$LOG_FILE"
}

log "========================================================"
log "  EVAL SWEEP"
log "  Date       : $(date)"
log "  Baseline   : $BASELINE_CKPT"
log "  FiLM       : $FILM_CKPT"
log "  OOD root   : $OOD_ROOT"
log "========================================================"

for test_set in "${TEST_SETS[@]}"; do
    metadata="$OOD_ROOT/$test_set/variants_metadata.json"

    if [ ! -f "$metadata" ]; then
        log ""
        log "[SKIP] No variants_metadata.json found at $metadata"
        continue
    fi

    # Get all variant names that are NOT the base robot
    variants=$(python3 - <<EOF
import json
with open("$metadata") as f:
    meta = json.load(f)
for name, info in meta.items():
    if info.get("set", "") != "base":
        print(name)
EOF
)

    if [ -z "$variants" ]; then
        log "[SKIP] No perturbed variants found in $metadata"
        continue
    fi

    log ""
    log "###################################################"
    log "  TEST SET: $test_set"
    log "###################################################"

    # ── BASELINE ──────────────────────────────────────────
    log ""
    log "========== BASELINE ================================"
    while IFS= read -r variant_name; do
        run_variant "BASELINE" "$BASELINE_CKPT" "--baseline" \
            "$test_set" "$variant_name" "$metadata"
    done <<< "$variants"

    # ── FiLM ──────────────────────────────────────────────
    log ""
    log "========== FILM ===================================="
    while IFS= read -r variant_name; do
        run_variant "FILM" "$FILM_CKPT" "" \
            "$test_set" "$variant_name" "$metadata"
    done <<< "$variants"

done

log ""
log "========================================================"
log "  SWEEP COMPLETE: $(date)"
log "  Log saved to: $LOG_FILE"
log "========================================================"