# Stitcher

Experiment to stitch images together without directional assumptions.  Takes a list of images as input and reduces list by iteratively stitching images together.  Process stops if images not deemed similiar enough (as deemed by an arbitrarily set keypoint match count).

## Example

### Input

<sub>*[`image_pieces`](image_pieces) is a directory containing images to be stitched*</sub>

```
python stitch.py -i image_pieces
```

### Output

![](readme/example_reduction.gif)
