#!/bin/bash

base_problems_dir="./problems"

echo "Starting all experiments..."

for category_dir in "$base_problems_dir"/*/; do
    category_dir="${category_dir%/}"
    category_name="$(basename "$category_dir")"

    echo ""
    echo "----------------------------------------------------"
    echo "Processing Category: $category_name"
    echo "----------------------------------------------------"

    if [[ "$category_name" == "Parametric_Equations" ]]; then

        for equation_dir in "$category_dir"/*/; do
            equation_dir="${equation_dir%/}"

            for problem_dir in "$equation_dir"/*/; do
                problem_dir="${problem_dir%/}"
                problem_name="$(basename "$equation_dir")/$(basename "$problem_dir")"

                initial_program_path="$problem_dir/initial_program.py"
                evaluator_path="$problem_dir/evaluator.py"
                config_path="$problem_dir/config.yaml"

                # Sanity checks
                if [[ ! -f "$initial_program_path" || ! -f "$evaluator_path" || ! -f "$config_path" ]]; then
                    echo "  [$problem_name] SKIPPING: Missing essential files."
                    continue
                fi

                echo "  Launching Parametric Sub-Problem: $problem_name"
                cmd="python ../openevolve-run.py \"$initial_program_path\" \"$evaluator_path\" --config \"$config_path\" --iterations 1000"
                eval $cmd &
            done
        done

    else
        for problem_dir in "$category_dir"/*/; do
            problem_dir="${problem_dir%/}"
            problem_name="$(basename "$problem_dir")"

            initial_program_path="$problem_dir/initial_program.py"
            evaluator_path="$problem_dir/evaluator.py"
            config_path="$problem_dir/config.yaml"

            # Sanity checks
            if [[ ! -f "$initial_program_path" ]]; then
                echo "  [$problem_name] SKIPPING: Initial program not found at $initial_program_path"
                continue
            fi
            if [[ ! -f "$evaluator_path" ]]; then
                echo "  [$problem_name] SKIPPING: Evaluator not found at $evaluator_path"
                continue
            fi
            if [[ ! -f "$config_path" ]]; then
                echo "  [$problem_name] SKIPPING: Config file not found at $config_path"
                continue
            fi

            echo "  Launching $category_name - $problem_name"
            cmd="python ../../openevolve-run.py \"$initial_program_path\" \"$evaluator_path\" --config \"$config_path\" --iterations 1000"
            eval $cmd &
        done
    fi
done

echo ""
echo "All experiment processes have been launched in the background."
echo "Waiting for all background processes to complete..."
wait
echo ""
echo "All experiments have completed."