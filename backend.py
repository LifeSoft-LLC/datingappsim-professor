import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import openpyxl  # for Excel export

##############################################################################
# 1) PRELOAD THE CSVs (PROFILES & PROBABILITY MATRICES)
##############################################################################
women_df = pd.read_csv("synthetic_women_profiles.csv")
men_df   = pd.read_csv("synthetic_men_profiles.csv")

prob_women_likes_men = pd.read_csv("probability_matrix_women_likes_men.csv", index_col=0)
prob_men_likes_women = pd.read_csv("probability_matrix_men_likes_women.csv", index_col=0)

# Create lookup dictionaries for profile info.
women_info = {row["WomanID"]: row for _, row in women_df.iterrows()}
men_info   = {row["ManID"]: row for _, row in men_df.iterrows()}

all_women_ids = list(women_info.keys())
all_men_ids   = list(men_info.keys())
all_user_ids  = all_women_ids + all_men_ids

##############################################################################
# 1.5) SELECT "JACK" AND "JILL" AS MIDDLE-PERFORMING PROFILES
##############################################################################
# For Jack, we choose the man whose average probability (from women liking men)
# is closest to the overall average among men.
man_avgs = prob_women_likes_men.mean(axis=0)
overall_man_avg = man_avgs.mean()
jack_id = (man_avgs - overall_man_avg).abs().idxmin()

# For Jill, we choose the woman whose average probability (from men liking women)
# is closest to the overall average among women.
woman_avgs = prob_men_likes_women.mean(axis=0)
overall_woman_avg = woman_avgs.mean()
jill_id = (woman_avgs - overall_woman_avg).abs().idxmin()

print(f"Selected Jack: {jack_id}, Selected Jill: {jill_id}")

##############################################################################
# 2) THE HINGE-LIKE SIMULATION FUNCTION WITH PERSISTENT UPDATING
##############################################################################
def run_dating_simulation(
    # Fixed parameters: num_days=3, daily_queue_size=5, random_seed=42
    num_days=3,
    daily_queue_size=5,
    weight_reciprocal=1.0,          # weight on probability that j likes i back
    weight_queue_penalty=0.5,       # penalty if candidate's incoming-like queue is long
    random_seed=42,
    export_trace=False,
    export_jack_jill_trace=False,
    show_match_plots=True,
    show_like_plots=True,
    plot_type="Bar Chart",          # Options: "Bar Chart" or "Histogram"
    summary_out=None,
    plot_out=None,
    trace_out=None,
    trace_jj_out=None
):
    """
    Runs a Tinder-style simulation in which, upon logging in,
    each user sees a single combined list of candidates. For each candidate:
    
      - If the candidate is already an incoming like (i.e. they previously liked the user),
        then the candidate’s score is defined as S̃₍ᵢⱼ₎ = Pᵢⱼ.
    
      - Otherwise (a fresh candidate), the score is:
            S₍ᵢⱼ₎ = Pᵢⱼ * 1/(1 + w_queue*Qⱼ) * (Pⱼᵢ)^(w_reciprocal)
        where Qⱼ is the number of pending likes for candidate j.
    
    The top daily_queue_size candidates (by score) are shown and processed.
    
    Extra metrics (unseen and stale unseen likes) and Jack & Jill trace export are also provided.
    
    NEW METRICS DEFINITIONS:
      - Unseen Likes: count of likes that were never seen by the recipient.
      - Stale Unseen Likes: count of unseen likes that were not sent on day 3.
    
    Plotting Options:
      - Match Plots: Displays matches per man/woman.
      - Like Plots: Displays likes sent per man/woman.
      - Plot Type: "Bar Chart" (individual counts) or "Histogram" (aggregated bins).
    """
    # Set seeds for reproducibility.
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    # Simulation state dictionaries.
    # For incoming likes, store tuples: (sender, sent_day)
    incoming_likes = {uid: [] for uid in all_user_ids}
    matches = {uid: set() for uid in all_user_ids}
    likes_sent = {uid: set() for uid in all_user_ids}
    daily_logs = []  # list of DataFrames (one per day)
    
    # Loop over simulation days.
    for day in range(1, num_days + 1):
        day_records = []
        login_order = all_user_ids.copy()
        random.shuffle(login_order)
        
        for user in login_order:
            # Determine candidate pool (opposite gender, not already matched).
            if user.startswith("W"):
                candidate_pool = [cid for cid in all_men_ids if cid not in matches[user]]
                get_prob = lambda cand: prob_women_likes_men.loc[user, cand]
                get_reciprocal = lambda cand: prob_men_likes_women.loc[cand, user]
            else:
                candidate_pool = [cid for cid in all_women_ids if cid not in matches[user]]
                get_prob = lambda cand: prob_men_likes_women.loc[user, cand]
                get_reciprocal = lambda cand: prob_women_likes_men.loc[cand, user]
            
            # Build a lookup from candidate -> sent_day for those who already liked user.
            incoming_for_user = {}
            for sender, sent_day in incoming_likes[user]:
                # If multiple incoming likes from the same candidate, take the earliest.
                if sender not in incoming_for_user or sent_day < incoming_for_user[sender]:
                    incoming_for_user[sender] = sent_day
            
            # Build a combined candidate list.
            candidate_info = []
            for cand in candidate_pool:
                if cand in incoming_for_user:
                    # Candidate already liked user: use score = P_ij.
                    score = get_prob(cand)
                    candidate_info.append({
                        "CandidateID": cand,
                        "Score": score,
                        "Source": "incoming",
                        "SentDay": incoming_for_user[cand]
                    })
                else:
                    # Fresh candidate.
                    q = len(incoming_likes[cand])  # pending likes for candidate
                    score = get_prob(cand) * (1/(1 + weight_queue_penalty * q)) * (get_reciprocal(cand) ** weight_reciprocal)
                    candidate_info.append({
                        "CandidateID": cand,
                        "Score": score,
                        "Source": "fresh",
                        "SentDay": day  # fresh likes are sent today
                    })
            
            # Sort the combined list by score (descending) and select top daily_queue_size.
            candidate_info_sorted = sorted(candidate_info, key=lambda x: x["Score"], reverse=True)
            selected_candidates = candidate_info_sorted[:daily_queue_size]
            
            # Process each selected candidate.
            for cand_record in selected_candidates:
                cand = cand_record["CandidateID"]
                source = cand_record["Source"]
                sent_day = cand_record["SentDay"]
                like_prob = get_prob(cand)
                roll = np.random.rand()
                decision = "Pass"
                match_formed = False
                
                if roll < like_prob:
                    decision = "Like"
                    # If candidate has already liked user, a match is formed.
                    if user in likes_sent[cand]:
                        match_formed = True
                        matches[user].add(cand)
                        matches[cand].add(user)
                    else:
                        likes_sent[user].add(cand)
                        # For fresh candidates, add this like to candidate's incoming likes.
                        if source == "fresh":
                            incoming_likes[cand].append((user, day))
                    # If the candidate came from the incoming list, remove that pending like.
                    if source == "incoming":
                        for idx, (s, sd) in enumerate(incoming_likes[user]):
                            if s == cand:
                                del incoming_likes[user][idx]
                                break
                delay = day - sent_day  # 0 if fresh; >0 if pending from an earlier day
                day_records.append({
                    "Day": day,
                    "UserID": user,
                    "CandidateID": cand,
                    "Score": cand_record["Score"],
                    "Source": source,
                    "LikeProbability": like_prob,
                    "RandomRoll": roll,
                    "Decision": decision,
                    "MatchFormed": match_formed,
                    "Delay": delay
                })
        daily_logs.append(pd.DataFrame(day_records))
    
    return daily_logs, matches, incoming_likes
