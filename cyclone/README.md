# Cyclones
The goal is to find comma cloud structures on satellite images with Mask RCNN [1]. The data for training was created manually (can share upon request). Results published in [2]. 

## Command line Usage
Train a new model starting from pre-trained COCO weights using `train` dataset 
```
python3 cyclone.py train --dataset=/path/to/dataset --subset=train --weight=coco
```

Resume training a model that you had trained earlier
```
python3 cyclone.py train --dataset=/path/to/dataset --weights=last
```

Generate masks with `test` images
```
python3 cyclone.py detect  --dataset=/path/to/dataset  --subset=test --weight=last
```

## Files

masks_cmp.ipynb - code for comparing results

## Links
[1] https://github.com/matterport/Mask_RCNN

[2] https://elibrary.ru/item.asp?id=44425939


