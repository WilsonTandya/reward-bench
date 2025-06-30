import argparse
import logging
import os
import sys
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.linear_model import LogisticRegression

from rewardbench import load_eval_dataset, save_to_hub
from rewardbench.constants import EXAMPLE_COUNTS, SUBSET_MAPPING
from rewardbench.generative import (
    API_MODEL_LIST,
    format_judge_answers,
    process_judgement,
    run_judge_pair,
)
from rewardbench.utils import calculate_scores_per_section


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, nargs="+", required=True)
    parser.add_argument("--num_threads", type=int, default=10)
    parser.add_argument("--pref_sets", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--do_not_save", action="store_true")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    logger.info(f"Running reward model on {args.model}")

    # Load dataset
    dataset, subsets = load_eval_dataset(
        core_set=not args.pref_sets,
        conv=None,
        custom_dialogue_formatting=True,
        tokenizer=None,
        logger=logger,
        keep_columns=["text_chosen", "text_rejected", "id"],
        max_turns=4,
    )

    ids = dataset["id"]
    dataset = dataset.remove_columns("id")
    if args.debug:
        dataset = dataset.select(range(10))
        subsets = subsets[:10]
        ids = ids[:10]

    # Run API inference
    def get_judgement(batch):
        prompt = batch["text_chosen"][0]["content"]
        answer_a = batch["text_chosen"]
        answer_b = batch["text_rejected"]
        is_shuffled = np.random.rand() > 0.5
        if is_shuffled:
            answer_a, answer_b = answer_b, answer_a
            winner_text = "B"
            loser_text = "A"
        else:
            winner_text = "A"
            loser_text = "B"
        winner, request, judgement = run_judge_pair(prompt, answer_a, answer_b, args.model[0])
        if isinstance(winner, list):
            winner = max(set(winner), key=winner.count)
        if winner == winner_text:
            return 1
        elif winner == loser_text:
            return 0
        else:
            return 0.5

    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = [executor.submit(get_judgement, x) for x in dataset]
        results = [f.result() for f in as_completed(futures)]

    out_dataset = dataset.add_column("results", results)
    out_dataset = out_dataset.add_column("subset", subsets)
    out_dataset = out_dataset.add_column("id", ids)

    # Compute lengths
    length_chosen = [len(" ".join([msg["content"] for msg in x])) for x in dataset["text_chosen"]]
    length_rejected = [len(" ".join([msg["content"] for msg in x])) for x in dataset["text_rejected"]]
    length_diff = [lc - lr for lc, lr in zip(length_chosen, length_rejected)]
    out_dataset = out_dataset.add_column("length_chosen", length_chosen)
    out_dataset = out_dataset.add_column("length_rejected", length_rejected)
    out_dataset = out_dataset.add_column("length_diff", length_diff)

    # Calculate raw winrates
    model_name = args.model[0] if isinstance(args.model, list) else args.model
    results_grouped = {"model": model_name, "model_type": "Generative RM"}
    for subset in np.unique(subsets):
        subset_dataset = out_dataset.filter(lambda ex: ex["subset"] == subset)
        winrate = np.mean([x for x in subset_dataset["results"] if x != 0.5])
        results_grouped[subset] = winrate
        print(f"{subset}: {winrate:.2%}")

    if not args.pref_sets:
        leaderboard = calculate_scores_per_section(EXAMPLE_COUNTS, SUBSET_MAPPING, results_grouped)
        print(leaderboard)

    # Inline Length-Controlled Evaluation
    print("\n--- Length-Controlled Winrate Estimation ---")
    df = pd.DataFrame({
        "win": out_dataset["results"],
        "length_diff": out_dataset["length_diff"],
        "instruction_id": out_dataset["id"]
    })
    df = df[df["win"] != 0.5]  # remove ties
    df = pd.get_dummies(df, columns=["instruction_id"], drop_first=True)
    X = df.drop(columns=["win"])
    y = df["win"]
    clf = LogisticRegression(max_iter=1000).fit(X, y)
    X_cf = X.copy()
    X_cf["length_diff"] = 0
    lc_preds = clf.predict_proba(X_cf)[:, 1]
    lc_winrate = 100 * lc_preds.mean()
    print(f"✅ Length-Controlled Winrate: {lc_winrate:.2f}%")
    raw_winrate = 100 * y.mean()
    print(f"📊 Raw Winrate (excluding ties): {raw_winrate:.2f}%")

    # Optionally save
    if not args.do_not_save:
        scores_dict = out_dataset.to_dict()
        scores_dict["model"] = model_name
        scores_dict["model_type"] = "Generative RM"
        save_to_hub(scores_dict, model_name, "eval-set-scores/", args.debug, local_only=False)


if __name__ == "__main__":
    main()
