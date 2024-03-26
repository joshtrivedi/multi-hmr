import pickle
import sys
import numpy as np

# read from command line argument
filename = sys.argv[1]

with open(filename, 'rb') as file:
    data = pickle.load(file)
    # print(data)
    # print(data.type)

def PCP(pred, gt, threshold):
    correct = np.sum(np.linalg.norm(pred-gt, axis = 1) < threshold)
    return correct / len(pred)

def PDJ(pred, gt, threshold):
    detected_joints = np.sum(np.linalg.norm(pred-gt, axis=1) < threshold)
    return detected_joints / (len(pred) * len(pred[0]))

def calculate_metrics(pred_keypoints, gt_keypoints, metric1, metric2, threshold):
    if metric1 == 'PCP':
        result1 = PCP(pred_keypoints, gt_keypoints, threshold)
    elif metric1 == 'PDJ':
        result1 = PDJ(pred_keypoints, gt_keypoints, threshold)

    if metric2 == 'PCP':
        result2 = PCP(pred_keypoints, gt_keypoints, threshold)
    elif metric2 == 'PDJ':
        result2 = PDJ(pred_keypoints, gt_keypoints, threshold)

    return result1, result2

with open(filename, 'rb') as file:
    allHumans = pickle.load(file)

threshold = 5
metric1 = "PDJ"
metric2 = "PCP"

results = []

for human in allHumans:
    pred_2d_keypoints = human['j2d_smplx']
    pred_3d_keypoints = human['j3d_smplx']
    # todo get groud truth
    gt_2d_keypoints = human['ground_truth_j2d']  
    gt_3d_keypoints = human['ground_truth_j3d']
    result1, result2 = calculate_metrics(pred_2d_keypoints, gt_2d_keypoints, metric1, metric2, threshold)
    results.append((result1, result2))

# Calculate mean of results
mean_result1 = np.mean([r[0] for r in results])
mean_result2 = np.mean([r[1] for r in results])

print(f"Mean {metric1}: {mean_result1}")
print(f"Mean {metric2}: {mean_result2}")
