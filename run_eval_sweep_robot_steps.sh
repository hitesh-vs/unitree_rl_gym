#!/bin/bash
# run_eval_sweep_steps.sh
# Runs physical step count sweep across all perturbed variants for both baseline and FiLM.
# Uses --find_best_init to find the best episode and report physical steps.
# Init .pt files are written to /tmp and overwritten each run — not kept.

cd /home/sviswasam/dr/unitree_rl_gym

BASELINE_CKPT="/home/sviswasam/dr/unitree_rl_gym/output_baseline_results2/Apr11_18-52-43/model_400.pt"
FILM_CKPT="/home/sviswasam/dr/unitree_rl_gym/output_film_results2/Apr11_18-53-28/model_400.pt"
XML_PATH="/home/sviswasam/dr/ModuMorph/modular/unitree_g1_actual/xml/g1_12dof_stripped.xml"
OOD_ROOT="/home/sviswasam/dr/unitree_rl_gym/resources/robots/g1_ood_test_sets2"
LOG_FILE="/home/sviswasam/dr/unitree_rl_gym/logs/step_sweep_$(date +%Y%m%d_%H%M%S).log"

# How many episodes to search through to find the best init
SEARCH_ROLLOUTS=850

TEST_SETS=(
    "armature_perturbed"
    "effort_perturbed"
    "mass_perturbed"
    "joint_range_perturbed"
    "all_perturbed"
    "damping_perturbed"
)

log() { echo "$@" | tee -a "$LOG_FILE"; }

run_variant_steps() {
    local model_label="$1"
    local ckpt="$2"
    local baseline_flag="$3"
    local test_set="$4"
    local variant_name="$5"
    local metadata="$6"

    log ""
    log "  [$model_label] $test_set / $variant_name"

    # Temp files — overwritten each call, not kept
    local tmp_init="/tmp/best_init_tmp.pt"
    local tmp_out="/tmp/traj_tmp.pkl"

    python deploy/deploy_mujoco/record_best_traj.py \
        --checkpoint "$ckpt" \
        --xml_path "$XML_PATH" \
        --variants_metadata "$metadata" \
        --variant_name "$variant_name" \
        --find_best_init \
        --init_search_rollouts "$SEARCH_ROLLOUTS" \
        --init_save "$tmp_init" \
        --out "$tmp_out" \
        --min_ep_filter 10 \
        $baseline_flag 2>&1 | grep -v "duplicate name" | grep -v "Geom with" | tee -a "$LOG_FILE"

    # Clean up tmp files
    rm -f "$tmp_init" "$tmp_out"
}

log "========================================================"
log "  PHYSICAL STEP COUNT SWEEP"
log "  Date          : $(date)"
log "  Baseline ckpt : $BASELINE_CKPT"
log "  FiLM ckpt     : $FILM_CKPT"
log "  Search per var: $SEARCH_ROLLOUTS episodes"
log "========================================================"

for test_set in "${TEST_SETS[@]}"; do
    metadata="$OOD_ROOT/$test_set/variants_metadata.json"

    if [ ! -f "$metadata" ]; then
        log "[SKIP] No metadata at $metadata"
        continue
    fi

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
        log "[SKIP] No perturbed variants in $metadata"
        continue
    fi

    log ""
    log "###################################################"
    log "  TEST SET: $test_set"
    log "###################################################"

    log ""
    log "===== BASELINE ====="
    while IFS= read -r variant_name; do
        run_variant_steps "BASELINE" "$BASELINE_CKPT" "--baseline" \
            "$test_set" "$variant_name" "$metadata"
    done <<< "$variants"

    log ""
    log "===== FILM ====="
    while IFS= read -r variant_name; do
        run_variant_steps "FILM" "$FILM_CKPT" "" \
            "$test_set" "$variant_name" "$metadata"
    done <<< "$variants"

done

log ""
log "========================================================"
log "  SWEEP COMPLETE: $(date)"
log "  Log: $LOG_FILE"
log "========================================================"
log ""
log "Key lines to grep from log:"
log "  grep 'Phase 1 done' $LOG_FILE"
log "  grep 'New best' $LOG_FILE"
log "  grep 'physical_steps' $LOG_FILE"