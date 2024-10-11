import argparse
import os
import pandas as pd
import numpy as np
import nibabel as nib
from conflunet.postprocess import remove_small_lesions_from_instance_segmentation
from conflunet.evaluation.utils import find_confluent_lesions
from conflunet.evaluation.semantic_segmentation import dice_metric, dice_norm_metric, intersection_over_union
from conflunet.evaluation.instance_segmentation import panoptic_quality, dice_per_tp
from conflunet.evaluation.detection import f_beta_score, recall, precision, pred_lesion_count, ref_lesion_count, DiC
from conflunet.evaluation.utils import match_instances


def compute_metrics(
        pred_path: str,
        ref_path: str,
        minimum_lesion_size: int = 14,
        test: bool = False
):
    # Check if prediction and reference folders exist
    if not os.path.exists(pred_path) or not os.path.exists(ref_path):
        print("Either prediction or reference path doesn't exist!")
        return

    metrics_dict = {"Subject_ID": [],
                    "File": [],
                    "DSC": [],
                    "nDSC": [],
                    "PQ": [],
                    "Fbeta": [],
                    "recall": [],
                    "precision": [],
                    "Dice_Per_TP": [],
                    "Pred_Lesion_Count": [],
                    "Ref_Lesion_Count": [],
                    "DiC": [],
                    "CLR": [],
                    "CLP":[],
                    "Dice_Per_TP_CL": [],
                    "CL_Count": []}
    all_pred_matches = {"Subject_ID": [], "Lesion_ID": [], "Ref_Lesion_ID_Match": [], "Volume_Pred": [],
                        "Volume_Ref": [], "DSC": []}
    all_ref_matches = {"Subject_ID": [], "Lesion_ID": [], "Pred_Lesion_ID_Match": [], "Volume_Ref": [],
                       "Volume_Pred": [], "DSC": []}

    #TODO change this to nnunet format
    dd = "test" if args.test else "val"
    ref_dir = os.path.join(ref_path, dd, "labels")

    for ref_file in sorted(os.listdir(ref_dir)):
        if ref_file.endswith("mask-instances.nii.gz"):
            print(ref_file)
            subj_id = ref_file.split("_ses")[0].split("sub-")[-1]  # Extracting subject ID
            pred_file = "sub-" + subj_id + "_ses-01_pred_instances.nii.gz"
            pred_file_path = os.path.join(args.pred_path, pred_file)

            if not os.path.exists(pred_file_path):
                pred_file_path = pred_file_path.replace('pred_instances', 'pred-instances')
            if not os.path.exists(pred_file_path):
                print(f"No prediction found for {pred_file}")
                continue

            ref_img = nib.load(os.path.join(ref_dir, ref_file))
            voxel_size = ref_img.header.get_zooms()
            ref_img = remove_small_lesions_from_instance_segmentation(ref_img.get_fdata(), voxel_size,
                                                                      l_min=args.minimum_lesion_size)
            pred_img = nib.load(pred_file_path).get_fdata()

            matched_pairs, unmatched_pred, unmatched_ref = match_instances(pred_img, ref_img)

            metrics_dict["Subject_ID"].append(subj_id)
            metrics_dict["File"].append(pred_file)

            ### Compute metrics ###
            # Dice score
            dsc = dice_metric((ref_img > 0).astype(np.uint8), (pred_img > 0).astype(np.uint8))
            metrics_dict["DSC"].append(dsc)
            
            # normalized Dice score
            ndsc = dice_norm_metric((ref_img > 0).astype(np.uint8), (pred_img > 0).astype(np.uint8))
            metrics_dict["nDSC"].append(ndsc)

            # PQ
            pq_val = panoptic_quality(pred=pred_img, ref=ref_img,
                                      matched_pairs=matched_pairs, unmatched_pred=unmatched_pred,
                                      unmatched_ref=unmatched_ref)
            metrics_dict["PQ"].append(pq_val)

            # F-beta score
            fbeta_val = f_beta_score(matched_pairs=matched_pairs, unmatched_pred=unmatched_pred,
                                     unmatched_ref=unmatched_ref)
            metrics_dict["Fbeta"].append(fbeta_val)

            # Recall and Precision
            recall_val = recall(matched_pairs=matched_pairs, unmatched_ref=unmatched_ref)
            metrics_dict["recall"].append(recall_val)
            precision_val = precision(matched_pairs=matched_pairs, unmatched_pred=unmatched_pred)
            metrics_dict["precision"].append(precision_val)

            # Additional metrics
            dice_scores = dice_per_tp(pred_img, ref_img, matched_pairs)
            avg_dice = sum(dice_scores) / len(dice_scores) if dice_scores else 0 # average per patient
            metrics_dict["Dice_Per_TP"].append(avg_dice)
            metrics_dict["Pred_Lesion_Count"].append(pred_lesion_count(pred_img))
            metrics_dict["Ref_Lesion_Count"].append(ref_lesion_count(ref_img))
            metrics_dict["DiC"].append(DiC(pred_img, ref_img))

            ### Confluent lesions Metrics ###
            confluents_ref_img = np.copy(ref_img)
            cl_ids = find_confluent_lesions(confluents_ref_img)

            # set all other ids to 0 in ref_img
            for id in np.unique(confluents_ref_img):
                if id not in cl_ids:
                    confluents_ref_img[confluents_ref_img == id] = 0

            matched_pairs_cl, unmatched_pred_cl, unmatched_ref_cl = match_instances(pred_img, confluents_ref_img)

            clm = len(cl_ids)
            metrics_dict["CL_Count"].append(clm)
            if clm == 0:
                metrics_dict["CLR"].append(np.nan)
                metrics_dict["CLP"].append(np.nan)
                metrics_dict["Dice_Per_TP_CL"].append(np.nan)
            else:
                clr = recall(matched_pairs=matched_pairs_cl, unmatched_ref=unmatched_ref_cl)
                clp = precision(matched_pairs=matched_pairs_cl, unmatched_pred=unmatched_pred_cl)
                metrics_dict["CLR"].append(clr)
                metrics_dict["CLP"].append(clp)

                dice_scores_cl = dice_per_tp(pred_img, confluents_ref_img, matched_pairs_cl)
                avg_dice_cl = sum(dice_scores_cl) / len(dice_scores_cl) if dice_scores_cl else 0
                metrics_dict["Dice_Per_TP_CL"].append(avg_dice_cl)

            ## Per lesion match & volume information ##
            # Store for every predicted lesion the potential match in the reference annotation,
            # along with both lesion volumes
            all_pred_labels, all_pred_counts = np.unique(pred_img[pred_img != 0], return_counts=True)
            pred_to_ref_matches = dict(matched_pairs)

            # If it was not already computed, compute the dice scores for each TP predicted lesion
            if not 'dice_scores' in locals():
                dice_scores = dice_per_tp(pred_img, ref_img, matched_pairs)

            for pid, volume_pred in zip(all_pred_labels, all_pred_counts):
                all_pred_matches["Subject_ID"].append(subj_id)
                all_pred_matches["Lesion_ID"].append(pid)

                if not (pid in unmatched_pred):
                    matched_ref_id = pred_to_ref_matches[pid]
                    volume_ref = sum((ref_img[ref_img == matched_ref_id] > 0).astype(np.uint8))
                    volume_ref *= np.prod(voxel_size)
                    this_pairs_dsc = dice_scores[matched_pairs.index((pid, matched_ref_id))]
                else:
                    matched_ref_id = None
                    volume_ref = None
                    this_pairs_dsc = None

                all_pred_matches["Ref_Lesion_ID_Match"].append(matched_ref_id)
                all_pred_matches["Volume_Pred"].append(volume_pred * np.prod(voxel_size))
                all_pred_matches["Volume_Ref"].append(volume_ref)
                all_pred_matches["DSC"].append(this_pairs_dsc)

            # Store for every lesion in the reference annotation the potential match in the predicted instance map,
            # along with both lesion volumes
            all_ref_labels, all_ref_counts = np.unique(ref_img[ref_img != 0], return_counts=True)
            ref_to_pred_matches = {v: k for k, v in dict(matched_pairs).items()}
            for rid, volume_ref in zip(all_ref_labels, all_ref_counts):
                all_ref_matches["Subject_ID"].append(subj_id)
                all_ref_matches["Lesion_ID"].append(rid)

                if not (rid in unmatched_ref):
                    matched_pred_id = ref_to_pred_matches[rid]
                    volume_pred = sum((pred_img[pred_img == matched_pred_id] > 0).astype(np.uint8))
                    volume_pred *= np.prod(voxel_size)
                    this_pairs_dsc = dice_scores[matched_pairs.index((matched_pred_id, rid))]
                else:
                    matched_pred_id = None
                    volume_pred = None
                    this_pairs_dsc = None

                all_ref_matches["Pred_Lesion_ID_Match"].append(matched_pred_id)
                all_ref_matches["Volume_Ref"].append(volume_ref * np.prod(voxel_size))
                all_ref_matches["Volume_Pred"].append(volume_pred)
                all_ref_matches["DSC"].append(this_pairs_dsc)

    model_name = os.path.basename(os.path.dirname(args.pred_path))

    # Convert dictionary to dataframe and save as CSV
    df = pd.DataFrame(metrics_dict)
    df.to_csv(os.path.join(args.pred_path, f"metrics_{model_name}_{dd}.csv"), index=False)

    # Save the per lesion matches
    df = pd.DataFrame(all_pred_matches)
    df.to_csv(os.path.join(args.pred_path, f"predicted_lesions_matches_{model_name}_{dd}.csv"), index=False)
    df = pd.DataFrame(all_ref_matches)
    df.to_csv(os.path.join(args.pred_path, f"reference_lesions_matches_{model_name}_{dd}.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute segmentation metrics.")
    parser.add_argument("--pred_path", required=True, help="Path to the directory with prediction files.")
    parser.add_argument("--ref_path", required=True,
                        help="Path to the directory with reference files (containing val/ and test/).")
    parser.add_argument("--minimum_lesion_size", default=14,
                        help="Minimum lesion size.")
    parser.add_argument("--test", action="store_true", help="Wether to use the test data or not. Default is val data.")

    args = parser.parse_args()
    compute_metrics(args)
