import re
import sys
import statistics

def summarize_logs(file_path):
    # Regex patterns
    header_re = re.compile(r"\[(BASELINE|FILM)\]\s+\S+\s+/\s+(\S+)")
    new_best_re = re.compile(r"New best:.*physical_steps=(\d+)")
    
    summaries = []
    current_entry = None

    with open(file_path, 'r') as f:
        for line in f:
            # Detect a new block
            header_match = header_re.search(line)
            if header_match:
                mode = header_match.group(1)
                variant = header_match.group(2)
                current_entry = {
                    "mode": mode,
                    "variant": variant,
                    "all_steps": []
                }
                summaries.append(current_entry)
                continue

            # Collect all "New best" physical_steps for this block
            if current_entry:
                best_match = new_best_re.search(line)
                if best_match:
                    current_entry["all_steps"].append(int(best_match.group(1)))

    def get_stats(steps_list):
        if not steps_list:
            return "N/A", "N/A"
        
        # Take the top 5 highest values
        top_5 = sorted(steps_list, reverse=True)[:5]
        
        avg = sum(top_5) / len(top_5)
        # Variance requires at least 2 points
        var = statistics.variance(top_5) if len(top_5) > 1 else 0.0
        
        return f"{avg:.2f}", f"{var:.2f}"

    # Table Header
    header = f"{'Variant':<25} | {'Type':<10} | {'Top 5 Avg':<12} | {'Top 5 Var':<10}"
    print(header)
    print("-" * len(header))

    # Print results
    for entry in summaries:
        avg, var = get_stats(entry["all_steps"])
        print(f"{entry['variant']:<25} | {entry['mode']:<10} | {avg:<12} | {var:<10}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        summarize_logs(sys.argv[1])
    else:
        print("Usage: python script.py <path_to_log_file>")