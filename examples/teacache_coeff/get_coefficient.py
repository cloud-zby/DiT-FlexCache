import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import json
from glob import glob


def load_data_from_json_files(base_dir, pattern):
    """
    Load data from JSON files with new format (name and items structure)
    """
    json_files = glob(os.path.join(base_dir, pattern))
    all_data = {}

    for json_file in json_files:
        try:
            with open(json_file, "r") as f:
                data = json.load(f)

            # Extract calibrate_data from all items
            if "items" in data:
                for item in data["items"]:
                    if "calibrate_data" in item:
                        calibrate_data = item["calibrate_data"]
                        # Extract data from calibrate_data
                        for key, values in calibrate_data.items():
                            if key not in all_data:
                                all_data[key] = []
                            all_data[key].append(values)
            else:
                raise ValueError(
                    f"Invalid format in {json_file}: missing 'items' field"
                )

        except Exception as e:
            print(f"Error loading {json_file}: {e}")

    if not all_data:
        raise ValueError(f"No valid JSON files found matching {pattern}")

    # Average across different files and prompts
    averaged_data = {}
    for key, value_lists in all_data.items():
        data_array = np.array(value_lists)
        averaged_data[key] = np.mean(data_array, axis=0)

    return averaged_data


def find_coefficient(
    result_dir=None,
    teacahce_coeffs=None,
    output_prefix="residual",
    order=4,
):
    try:
        has_uncond = len(glob(os.path.join(result_dir, "uncond*.json"))) > 0

        # Load conditional data
        cond_data = load_data_from_json_files(result_dir, "cond*.json")

        # Load unconditional data if exists and combine
        if has_uncond:
            uncond_data = load_data_from_json_files(result_dir, "uncond*.json")

            # Combine conditional and unconditional data
            combined_data = {}
            for key in cond_data.keys():
                if key in uncond_data:
                    # Average the conditional and unconditional data
                    combined_data[key] = (cond_data[key] + uncond_data[key]) / 2
                else:
                    combined_data[key] = cond_data[key]
        else:
            combined_data = cond_data

        # Extract the required data directly
        mod_input_diff = combined_data.get("mod_input_diffs")
        output_diff = combined_data.get("res_output_diffs")

        if mod_input_diff is None or output_diff is None:
            raise ValueError(
                "Required data (mod_input_diffs, res_output_diffs) not found in JSON files"
            )

        # Create dummy data for missing fields if needed
        input_diff = combined_data.get("input_diffs")
        time_emb_diff = combined_data.get("time_emb_diffs")

    except Exception as e:
        print(f"JSON loading error - {e}")
        return

    try:
        # Data is already averaged numpy arrays from JSON
        x = mod_input_diff
        y = output_diff
        n = input_diff
        t = time_emb_diff

        if x.shape != y.shape:
            raise ValueError(f"shape not match - input: {x.shape}, output: {y.shape}")

        steps = np.arange(len(x))

    except Exception as e:
        print(f"error - {e}")
        return

    os.makedirs(os.path.dirname(output_prefix) or ".", exist_ok=True)
    plt.figure(figsize=(15, 12))

    # ======== figure1 ==========
    plt.subplot(2, 2, 1)
    plt.plot(steps, x, "b-", label="Mod Input Diff (Modulated)", linewidth=2)
    plt.plot(steps, y, "g-", label="Output Diff (Residual)", linewidth=2)
    plt.plot(steps, n, "y-", label="Noisy Input Diff ", linewidth=2)
    plt.plot(steps, t, "r-", label="Time Emb Diff ", linewidth=2)
    plt.xlabel("Diffusion Step", fontsize=12)
    plt.ylabel("Difference Value", fontsize=12)
    plt.title("Diffusion Process: Input vs Output Differences", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xticks(np.arange(0, len(x) + 1, 5))

    # ======== figure2 =========
    plt.subplot(2, 2, 2)
    coefficients = np.polyfit(x, y, order)
    rescale_func = np.poly1d(coefficients)
    ypred = rescale_func(x)

    plt.plot(np.log(x), np.log(y), "g*", markersize=8, label="Actual (log)")
    plt.plot(np.log(x), np.log(ypred), "b.", markersize=10, label="Fitted (log)")
    ax = plt.gca()
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    diag_min = max(xmin, ymin)
    diag_max = min(xmax, ymax)
    plt.plot(
        [diag_min, diag_max],
        [diag_min, diag_max],
        "k-",
        linewidth=1.5,
        alpha=0.7,
        label="x=y (ideal)",
    )
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    plt.xlabel("log(Input Diff)", fontsize=12)
    plt.ylabel("log(Output Diff)", fontsize=12)
    plt.ylim(np.floor(np.min(np.log(y))) - 0.5, 1)
    plt.title(f"{order}th Order Polynomial Fit (Custom)", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)

    # ======== figure3 =========
    if teacahce_coeffs is not None:
        plt.subplot(2, 2, 3)
        rescale_func_tea = np.poly1d(teacahce_coeffs)
        ypred_tea = rescale_func_tea(x)

        plt.plot(np.log(x), np.log(y), "g*", markersize=8, label="Actual (log)")
        plt.plot(
            np.log(x), np.log(ypred_tea), "r.", markersize=10, label="Fitted (log)"
        )

        ax = plt.gca()
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        diag_min = max(xmin, ymin)
        diag_max = min(xmax, ymax)
        plt.plot(
            [diag_min, diag_max],
            [diag_min, diag_max],
            "k-",
            linewidth=1.5,
            alpha=0.7,
            label="x=y (Ideal)",
        )

        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.xlabel("log(Input Diff)", fontsize=12)
        plt.ylabel("log(Output Diff)", fontsize=12)
        plt.ylim(np.floor(np.min(np.log(y))) - 0.5, 1)
        plt.title(f"{order}th Order Polynomial Fit (Teacahce)", fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    output_image = f"{output_prefix}analysis.png"
    plt.savefig(output_image, dpi=300, bbox_inches="tight")
    print(f"\nthe results have been saved: {output_image}")

    print("\n== result ===")
    print("JSON files processed")
    print(f"steps: {len(x)}")
    print("\ncoefficients:")
    for i, coeff in enumerate(coefficients):
        print(f"x^{len(coefficients)-1-i}: {coeff:.8e}")

    ss_res = np.sum((y - ypred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    print(f"\nR^2: {r_squared:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="分析输入输出差异并进行多项式拟合")
    parser.add_argument(
        "--result_dir",
        type=str,
        default="/nvfile-heatstorage/zgf/CODE/teleai_pipe/work_dirs/teacache/wan14b_i2v",
    )
    parser.add_argument(
        "--order",
        type=int,
        default=4,
    )
    args = parser.parse_args()
    teacahce_coeffs = []
    # # TeaCache预设系数
    # teacahce_coeffs = [
    #     7.33226126e02,
    #     -4.01131952e02,
    #     6.75869174e01,
    #     -3.14987800e00,
    #     9.61237896e-02,
    # ]

    find_coefficient(
        result_dir=args.result_dir,
        teacahce_coeffs=teacahce_coeffs,
        output_prefix=args.result_dir,
        order=args.order,
    )
