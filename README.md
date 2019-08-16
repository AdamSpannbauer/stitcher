# Stitcher

Experiment to stitch images together without directional assumptions.  Takes a list of images as input and reduces list by iteratively stitching images together.  Process stops if images not deemed similiar enough (as deemed by an arbitrarily set keypoint match count).

## Example

### Input


```
python stitch.py -i image_pieces
```

<sub>*[`image_pieces`](image_pieces) is a directory containing images to be stitched*</sub>


### Output

<img src='readme/example_reduction.gif' align='center' width='75%'>
