import joblib

output = joblib.load('/home/user/ply/data/five_sts/keypoints_3d/vibe/side/multi_person/cs-side-225/vibe_output.pkl')

print(output.keys())

for k,v in output[1].items(): print(k,v.shape) 
