# Overshooting top
The goal is to find overshooting top on satellite IR images with Mask RCNN

## Command line Usage
Train a new model starting from pre-trained COCO weights using `train` dataset 
```
python3 storm.py train --dataset=/path/to/dataset --subset=train --weight=coco
```

Resume training a model that you had trained earlier
```
python3 balloon.py train --dataset=/path/to/dataset --weights=last
```

Generate masks with `test` images
```
python3 storm.py detect  --dataset=/path/to/dataset  --subset=test --weight=last
```

## Links
https://github.com/matterport/Mask_RCNN
