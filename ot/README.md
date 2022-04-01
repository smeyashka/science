# Overshooting top
The goal is to find overshooting top on satellite IR images with Mask RCNN [1]. The data for training was created using the Bedkа algorithm [2], after which it was checked manually (can share upon request). Results published in [3].

## Command line Usage
Train a new model starting from pre-trained COCO weights using `train` dataset 
```
python3 storm.py train --dataset=/path/to/dataset --subset=train --weight=coco
```

Resume training a model that you had trained earlier
```
python3 storm.py train --dataset=/path/to/dataset --weights=last
```

Generate masks with `test` images
```
python3 storm.py detect  --dataset=/path/to/dataset  --subset=test --weight=last
```

## Files
ot_goes_bedka_one_part.ipynb - experiments to create input data for the neural network

masks_cmp.ipynb - code for comparing results

## Links
[1] https://github.com/matterport/Mask_RCNN

[2] https://journals.ametsoc.org/view/journals/apme/49/2/2009jamc2286.1.xml

[3] https://www.elibrary.ru/item.asp?id=44052060

