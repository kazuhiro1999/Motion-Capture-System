

def to_dict(timestamp, keypoints3d, keys, type):
    data = {
        "Type":type, 
        "TimeStamp": timestamp, 
        "Bones":[]
        }

    if keypoints3d is None:
        return data 
    
    for key in keys:
        bone = {
            "Name": key,
            "Position":{
                "x": float(keypoints3d[keys[key],0]),
                "y": float(keypoints3d[keys[key],1]),
                "z": float(keypoints3d[keys[key],2]),
            }
        }
        data['Bones'].append(bone)  
    return data